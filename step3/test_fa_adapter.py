"""Exercise the actual FA slicing method without loading GPU/MFA dependencies."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf


SOURCE = Path(__file__).resolve().parents[1]/'models/qwen3-fa-mfa/final_script/adapter.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
adapter = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Adapter')
method = next(n for n in adapter.body if isinstance(n, ast.FunctionDef) and n.name == '_chunk_audio')
method.decorator_list = []
module = ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
namespace = dict(Path=Path,sf=sf)
exec(compile(module,str(SOURCE),'exec'),namespace)
slice_audio = namespace['_chunk_audio']


class AdapterRegressionTests(unittest.TestCase):
    def test_stereo_selection_and_absolute_bounds(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'source.wav'
            sf.write(source,np.column_stack([np.full(8000,.125),np.full(8000,.5)]),8000)
            row=dict(id='chunk',audio=str(source),duration=.25,channel=1,
                     audio_start_sample=2000,audio_end_sample=4000)
            prepared=slice_audio(row,root)
            wave,rate=sf.read(prepared['audio'])
            self.assertEqual((wave.ndim,len(wave),rate),(1,2000,8000))
            self.assertAlmostEqual(float(wave.mean()),.5,places=4)

    def test_precut_mono_preserves_logical_channel(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'source.wav'
            sf.write(source,np.zeros(2000),8000)
            row=dict(id='chunk',audio=str(source),duration=.25,channel=1,
                     audio_start_sample=80000,audio_end_sample=82000)
            self.assertEqual(slice_audio(row,root),row)

    def test_full_bounds_cannot_bypass_duration_validation(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'source.wav'
            sf.write(source,np.zeros(8000),8000)
            row=dict(id='chunk',audio=str(source),duration=.25,channel=0,
                     audio_start_sample=0,audio_end_sample=8000)
            with self.assertRaisesRegex(ValueError,'duration mismatch'):
                slice_audio(row,root)


if __name__=='__main__':
    unittest.main()
