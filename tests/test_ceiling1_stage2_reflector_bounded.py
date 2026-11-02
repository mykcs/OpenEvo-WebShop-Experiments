import importlib.util, pathlib, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
P=ROOT/'scripts/openevo_webshop/ceiling1_stage2_reflector.py'
spec=importlib.util.spec_from_file_location('reflector',P)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def evidence():
    summaries=[]
    for i in range(16):
        summaries.append({'task_identity':f'task-{i}','best_score':1.0,'worst_score':0.0,'score_values':[0,1],'clean_success_count':1,'failure_signatures':['invalid_action'],'best_examples':[{'schedule_position':i,'score':1.0,'clean_exact_success':True,'sequence':'search>click>buy','actions':['search[x]','click[y]','click[Buy Now]'],'failure_signature':None,'first_prompt_excerpt':'x'*2000}]})
    patterns=[{'pattern':'failure:invalid_action','kind':'failure_pattern','current_task_support':16,'total_task_support':100,'task_ids':['very-long-id']*100}]
    gate={'eligible':True,'supporting_patterns':patterns}
    return {'round_index':4,'attempt_count':128,'task_identity_count':16,'summaries':summaries,'gates':{'text_memory':{'eligible':True,'clean_success_count':1,'contrast_task_count':2,'repeated_failure_patterns':['invalid_action']},'skill_bundle':gate,'agent_system':gate}}

class BoundedReflector(unittest.TestCase):
    def test_projection_is_fixed_and_bounded(self):
        e=evidence(); p=m.bounded_projection('skill_bundle',e)
        self.assertEqual(len(p['task_summaries']),16)
        self.assertNotIn('task_ids',repr(p['gate']))
        self.assertEqual(len(p['task_summaries'][0]['examples']),1)
        self.assertEqual(len(p['task_summaries'][0]['examples'][0]['task_prompt_excerpt']),400)
        self.assertFalse(p['projection_rules']['outcome_adaptive_projection'])
    def test_repair_reuses_same_projection(self):
        e=evidence(); a,p1=m.render('agent_system',e,'prior','memory','skill',False); b,p2=m.render('agent_system',e,'prior','memory','skill',True)
        self.assertEqual(p1,p2); self.assertIn('Fixed one-time repair',b); self.assertNotIn('Previous candidate',b)
        src=P.read_text(); self.assertIn('n+MAX_NEW_TOKENS>context_window',src); self.assertIn("'silent_truncation_allowed':False",src)

if __name__=='__main__': unittest.main()
