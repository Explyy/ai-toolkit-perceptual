"""Synthetic protobuf metadata only; no model weights or inference."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location('recognition',Path(__file__).resolve().parents[2]/'extensions_built_in/dataset_studio_analysis/recognition.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def varint(n):
    b=bytearray()
    while n>127:b.append((n&127)|128);n>>=7
    b.append(n);return bytes(b)
def field(n,b):return varint((n<<3)|2)+varint(len(b))+b
def model(names):return field(7,b''.join(field(1,field(3,n.encode())) for n in names)+field(5,b'synthetic tensor bytes to skip'))
class Normalization(unittest.TestCase):
    def check(self,names,expected):
        with tempfile.TemporaryDirectory() as t:
            file=Path(t)/'synthetic.onnx';file.write_bytes(model(names))
            self.assertEqual(r.recognition_normalization(file),expected)
    def test_standard_recognition_input(self):self.check(['Conv_0','PRelu_1'],{'mean':127.5,'std':127.5,'embedded':False})
    def test_embedded_normalization_and_first_eight_boundary(self):
        self.check(['Sub_0','Mul_1','Conv_2'],{'mean':0.0,'std':1.0,'embedded':True})
        self.check(['_minus0','_mul0'],{'mean':0.0,'std':1.0,'embedded':True})
        self.check(['Conv']*8+['Sub','Mul'],{'mean':127.5,'std':127.5,'embedded':False})
    def test_truncated_or_empty_graph_refused(self):
        with tempfile.TemporaryDirectory() as t:
            file=Path(t)/'synthetic.onnx'
            for data in [b'',field(7,b''),model(['Conv'])[:-5],b'\x3a\xff']:
                file.write_bytes(data)
                with self.assertRaises(ValueError):r.recognition_normalization(file)
if __name__=='__main__':unittest.main()
