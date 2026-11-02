"""CPU-only stage admission; execute the producer's actual pure validator."""
import ast,unittest
from pathlib import Path
source=Path(__file__).resolve().parents[1]/'scripts/openevo_webshop/q09_stage_a_eval.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='validate_stage_identity')
ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
validate=ns['validate_stage_identity']
class StageIdentity(unittest.TestCase):
 def test_both_fresh_endpoints_and_old_development_remain_distinct(self):
  for stage,treatment,stream,status in [('common-start-confirmation','SHARED_START','shared-common-start','FROZEN_PREOUTCOME'),('confirmation-40','ACTIVE','q09-stream-1','FROZEN_PREOUTCOME'),('common-start','SHARED_START','shared-common-start','FROZEN_EXISTING_DEVELOPMENT_PANEL'),('development-40','ACTIVE','q09-stream-1','FROZEN_EXISTING_DEVELOPMENT_PANEL')]:
   self.assertEqual(validate(dict(evaluation_stage=stage,treatment=treatment,stream_id=stream),status),stage)
 def test_fresh_cannot_be_development_or_relabelled_active(self):
  cfg=dict(evaluation_stage='common-start-confirmation',treatment='SHARED_START',stream_id='shared-common-start')
  for key,value in [('treatment','ACTIVE'),('stream_id','q09-stream-1'),('evaluation_stage','common-start')]:
   with self.subTest(key=key),self.assertRaises(SystemExit):validate(dict(cfg,**{key:value}),'FROZEN_PREOUTCOME')
  with self.assertRaises(SystemExit):validate(cfg,'FROZEN_EXISTING_DEVELOPMENT_PANEL')
 def test_unrecognized_stage_fails_closed(self):
  with self.assertRaises(SystemExit):validate(dict(evaluation_stage='fresh-anything'),'FROZEN_PREOUTCOME')
if __name__=='__main__':unittest.main()
