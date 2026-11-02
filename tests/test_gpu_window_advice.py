"""CPU-only negative controls for expiring GPU-grant advisory."""
import copy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.openevo_webshop.gpu_window_advice import PlanError, advise

ROOT=Path(__file__).resolve().parents[1]
AT="2026-10-14T08:00:00+08:00"

def window(gpu,start="08:00",end="12:00",buffer=10):
    return {"gpu_index":gpu,
        "starts_at":f"2026-10-14T{start}:00+08:00",
        "handoff_at":f"2026-10-14T{end}:00+08:00",
        "handoff_buffer_minutes":buffer,
        "authorization_ref":"SYNTHETIC_TEST_NO_AUTHORITY"}

def job(name,priority=1,cards=(6,),count=1,p95=20,units=1):
    return {"job_id":name,"priority":priority,"gpu_count":count,
        "eligible_gpu_indices":list(cards),"units":units,
        "unit_p95_minutes":p95,"startup_minutes":5,"release_minutes":5,
        "ready_for_planning":True,"depends_on":[],
        "allow_gpu_rebind_between_units":False,
        "science_ref":"SYNTHETIC_NOT_FROZEN",
        "estimate_ref":"SYNTHETIC_NOT_MEASURED"}

def plan(windows=None,jobs=None,cap=2):
    return {"contract_version":"openevo.gpu-window-advice.202610130200",
        "execution_mode":"simulation-only","timezone":"Asia/Singapore",
        "max_parallel_gpus":cap,
        "gpu_windows":windows if windows is not None else [window(6),window(7)],
        "jobs":jobs if jobs is not None else []}

class TimeWindowAdvisoryTests(unittest.TestCase):
    def test_two_card_training_starts_now_and_short_backfills_before_handoff(self):
        demo=json.loads((ROOT/"templates/gpu-window-advice-202610130200.example.json").read_text())
        res=advise(demo,at=AT)
        self.assertEqual(res["status"],"DRY_RUN_ONLY_NOT_AUTHORITY")
        self.assertEqual([x["job_id"] for x in res["scheduled_units"][:2]],
            ["example-primary-two-card-training"]*2)
        self.assertEqual(res["scheduled_units"][0]["gpu_indices"],[6,7])
        self.assertEqual(res["scheduled_units"][0]["planned_start_at"],AT)
        handback=next(w for w in res["handoffs"] if w["gpu_index"]==6)
        self.assertEqual(handback["handoff_to"],"EXAMPLE_lab_colleague")
        short=next(x for x in res["scheduled_units"] if x["job_id"]=="example-gpu6-short-backfill")
        self.assertEqual(short["planned_released_at"],"2026-10-14T10:40:00+08:00")
        for item in res["scheduled_units"]:
            finish=datetime.fromisoformat(item["planned_released_at"])
            for gpu in item["gpu_indices"]:
                grant=next(w for w in res["handoffs"] if w["gpu_index"]==gpu)
                self.assertLessEqual(finish,datetime.fromisoformat(grant["latest_scheduling_cutoff"]))
        self.assertEqual(res["projected_usable_gpu_minutes"],
            res["projected_busy_gpu_minutes"]+res["projected_idle_gpu_minutes"])

    def test_oversized_or_out_of_window_work_stays_queued(self):
        x=advise(plan([window(6,end="09:00")],[job("long",cards=(6,),p95=55)],cap=1),at=AT)
        self.assertEqual(x["scheduled_units"],[])
        self.assertEqual(x["pending_jobs"][0]["reason"],"WINDOW_OR_SAFE_BOUNDARY_UNAVAILABLE")
        y=advise(plan([window(6,end="09:00")],[job("A",cards=(6,))],cap=1),
            at="2026-10-14T10:00:00+08:00")
        self.assertFalse(y["scheduled_units"])
        self.assertEqual(y["projected_usable_gpu_minutes"],0)

    def test_future_high_priority_gang_protected_from_long_backfill(self):
        x=plan([window(6),window(7,start="08:30")],[
            job("gang",priority=1,cards=(6,7),count=2),
            job("short",priority=2,cards=(6,),p95=10),
            job("long",priority=3,cards=(6,),p95=40)])
        events={i["job_id"]:i for i in advise(x,at=AT)["scheduled_units"]}
        self.assertEqual(events["short"]["planned_start_at"],AT)
        self.assertEqual(events["short"]["planned_released_at"],"2026-10-14T08:20:00+08:00")
        self.assertEqual(events["gang"]["planned_start_at"],"2026-10-14T08:30:00+08:00")
        self.assertEqual(events["gang"]["gpu_indices"],[6,7])
        self.assertEqual(events["long"]["planned_start_at"],"2026-10-14T09:00:00+08:00")

    def test_dependency_release_reserves_earliest_gang_start_against_long_backfill(self):
        # A finishes on GPU7 at 08:20. Its dependent two-GPU B should start
        # then, rather than allowing unrelated lower-priority C to delay it.
        a=job("A",priority=1,cards=(7,),p95=10)
        b=job("B",priority=1,cards=(6,7),count=2,p95=10)
        b["depends_on"]=["A"]
        c=job("C",priority=2,cards=(6,),p95=30)
        result=advise(plan(jobs=[a,b,c]),at=AT)
        events={entry["job_id"]:entry for entry in result["scheduled_units"]}
        self.assertEqual(events["A"]["planned_released_at"],"2026-10-14T08:20:00+08:00")
        self.assertEqual(events["B"]["planned_start_at"],"2026-10-14T08:20:00+08:00")
        self.assertEqual(events["B"]["gpu_indices"],[6,7])
        self.assertGreaterEqual(
            datetime.fromisoformat(events["C"]["planned_start_at"]),
            datetime.fromisoformat(events["B"]["planned_released_at"]))

    def test_short_backfill_still_fits_before_known_dependency_release(self):
        a=job("A",priority=1,cards=(7,),p95=10)
        b=job("B",priority=1,cards=(6,7),count=2,p95=10)
        b["depends_on"]=["A"]
        short=job("short",priority=2,cards=(6,),p95=10)
        result=advise(plan(jobs=[a,b,short]),at=AT)
        events={entry["job_id"]:entry for entry in result["scheduled_units"]}
        self.assertEqual(events["short"]["planned_start_at"],AT)
        self.assertEqual(events["short"]["planned_released_at"],"2026-10-14T08:20:00+08:00")
        self.assertEqual(events["B"]["planned_start_at"],"2026-10-14T08:20:00+08:00")

    def test_parallel_cap_does_not_overbook_disjoint_cards(self):
        rows=advise(plan(jobs=[job("first",cards=(6,)),
            job("second",priority=2,cards=(7,))],cap=1),at=AT)["scheduled_units"]
        self.assertEqual(rows[0]["planned_start_at"],AT)
        self.assertGreaterEqual(datetime.fromisoformat(rows[1]["planned_start_at"]),
            datetime.fromisoformat(rows[0]["planned_released_at"]))

    def test_capacity_metrics_respect_parallel_card_limit(self):
        base=plan(jobs=[],cap=1)
        r=advise(base,at=AT)
        self.assertEqual(r["candidate_grant_gpu_minutes_before_concurrency_cap"],460)
        self.assertEqual(r["projected_usable_gpu_minutes"],230)
        self.assertEqual(r["projected_idle_gpu_minutes"],230)
        self.assertEqual(r["projected_busy_gpu_minutes"],0)

    def test_units_not_migrated_without_scientific_permission(self):
        data=plan([window(6,end="09:00"),window(7)],
            [job("work",cards=(6,7),units=2,p95=30)])
        fixed=advise(data,at=AT)
        self.assertEqual(len(fixed["scheduled_units"]),1)
        self.assertEqual(fixed["pending_jobs"][0]["units_pending"],1)
        migratable=copy.deepcopy(data)
        migratable["jobs"][0]["allow_gpu_rebind_between_units"]=True
        result=advise(migratable,at=AT)
        self.assertEqual(len(result["scheduled_units"]),2)
        self.assertEqual(result["scheduled_units"][1]["gpu_indices"],[7])

    def test_unready_or_unfinished_dependencies_cannot_start(self):
        parent=job("parent");parent["ready_for_planning"]=False
        child=job("child",priority=2,cards=(7,))
        child["depends_on"]=["parent"]
        r=advise(plan(jobs=[parent,child]),at=AT)
        self.assertFalse(r["scheduled_units"])
        self.assertEqual({x["reason"] for x in r["pending_jobs"]},
            {"NOT_PLANNING_READY","DEPENDENCY_PENDING"})

    def test_completed_dependency_is_respected(self):
        a=job("A",cards=(6,),p95=40)
        b=job("B",priority=2,cards=(7,));b["depends_on"]=["A"]
        events={x["job_id"]:x for x in advise(plan(jobs=[b,a]),at=AT)["scheduled_units"]}
        self.assertGreaterEqual(datetime.fromisoformat(events["B"]["planned_start_at"]),
                                datetime.fromisoformat(events["A"]["planned_released_at"]))

    def test_bad_grants_or_uncertain_training_inputs_fail_closed(self):
        base=plan(jobs=[job("A")])
        variants=[]
        p=copy.deepcopy(base);p["gpu_windows"][1]["gpu_index"]=6;variants.append(p)
        p=copy.deepcopy(base);p["max_parallel_gpus"]=3;variants.append(p)
        p=copy.deepcopy(base);p["jobs"][0]["eligible_gpu_indices"]=[5];variants.append(p)
        p=copy.deepcopy(base);p["execution_mode"]="execute";variants.append(p)
        p=copy.deepcopy(base);p["gpu_windows"][0]["handoff_buffer_minutes"]=0;variants.append(p)
        p=copy.deepcopy(base);p["jobs"][0]["unit_p95_minutes"]=0;variants.append(p)
        p=copy.deepcopy(base);p["jobs"][0]["release_minutes"]=0;variants.append(p)
        p=copy.deepcopy(base);p["jobs"][0]["priority"]=0;variants.append(p)
        p=copy.deepcopy(base);p["jobs"][0]["ready_for_planning"]=1;variants.append(p)
        p=copy.deepcopy(base);p["jobs"][0]["science_ref"]="";variants.append(p)
        p=copy.deepcopy(base);p["jobs"][0]["depends_on"]=["ghost"];variants.append(p)
        p=copy.deepcopy(base);p["jobs"][0]["depends_on"]=["A"];variants.append(p)
        p=copy.deepcopy(base);p["gpu_windows"][0]["handoff_at"]="2026-10-14T08:03:00+08:00";variants.append(p)
        p=copy.deepcopy(base);p["gpu_windows"][0]["starts_at"]="2026-10-14T08:00:00";variants.append(p)
        p=copy.deepcopy(base);p["gpu_windows"][0]["starts_at"]="2026-10-14T08:00:00+00:00";variants.append(p)
        for bad in variants:
            with self.subTest(bad=bad),self.assertRaises(PlanError):
                advise(bad,at=AT)

    def test_cyclic_dependencies_are_rejected(self):
        x=job("A");y=job("B",cards=(7,),priority=2)
        x["depends_on"]=["B"];y["depends_on"]=["A"]
        with self.assertRaisesRegex(PlanError,"cycle"):
            advise(plan(jobs=[x,y]),at=AT)

    def test_canonical_oe_has_cpu_only_gpu_window_preview(self):
        sample=ROOT/"templates/gpu-window-advice-202610130200.example.json"
        command=[sys.executable,"scripts/oe","gpu-window-plan","--plan",str(sample),"--at",AT]
        completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stderr)
        payload=json.loads(completed.stdout)
        self.assertEqual(payload["status"],"DRY_RUN_ONLY_NOT_AUTHORITY")
        source=(ROOT/"scripts/oe").read_text(encoding="utf-8")
        self.assertIn('if raw and raw[0] == "gpu-window-plan":',source)
        self.assertIn("Formal launches remain run-campaign only",source)

    def test_cpu_only_cli_and_no_mutating_controls(self):
        source=(ROOT/"scripts/openevo_webshop/gpu_window_advice.py").read_text()
        for verb in ("import docker","import ray","nvidia-smi","docker run",
                     "subprocess.","os.system","os.kill","cuda"):
            self.assertNotIn(verb,source)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"plan.json"
            path.write_text(json.dumps(plan([window(6)],[job("A",cards=(6,))],cap=1)))
            base=[sys.executable,"-m","scripts.openevo_webshop.gpu_window_advice",
                  "--plan",str(path),"--at",AT]
            ok=subprocess.run(base,cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(ok.returncode,0,ok.stderr)
            self.assertEqual(json.loads(ok.stdout)["scheduled_units"][0]["job_id"],"A")
            data=json.loads(path.read_text())
            data["execution_mode"]="execute";path.write_text(json.dumps(data))
            blocked=subprocess.run(base,cwd=ROOT,text=True,capture_output=True)
            self.assertNotEqual(blocked.returncode,0)
            self.assertIn("simulation-only",blocked.stderr)


if __name__=="__main__":
    unittest.main()
