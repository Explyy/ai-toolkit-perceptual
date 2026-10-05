"""Read ONNX graph node names without tensors, imports or model execution.
Normalization rule matches InsightFace v0.7 ArcFace adapter (MIT code):
https://github.com/deepinsight/insightface/blob/v0.7/python-package/insightface/model_zoo/arcface_onnx.py
No InsightFace weights or runtime package are included.
"""
from pathlib import Path

def _varint(stream):
    value=0
    for i in range(10):
        byte=stream.read(1)
        if not byte: raise ValueError('Truncated ONNX metadata')
        value|=(byte[0]&127)<<(7*i)
        if byte[0]<128: return value
    raise ValueError('Invalid ONNX metadata integer')

def _fields(stream,end):
    while stream.tell()<end:
        tag=_varint(stream);field,wire=tag>>3,tag&7
        if field==0: raise ValueError('Invalid ONNX metadata tag')
        if wire==0:
            _varint(stream)
        elif wire in (1,5):
            stream.seek(8 if wire==1 else 4,1)
        elif wire==2:
            length=_varint(stream);start=stream.tell()
            if start+length>end: raise ValueError('Truncated ONNX metadata field')
            yield field,start,length
            stream.seek(start+length)
        else: raise ValueError('Unsupported ONNX metadata wire format')
        if stream.tell()>end: raise ValueError('Truncated ONNX metadata scalar')

def recognition_normalization(file:Path):
    names=[]
    with file.open('rb') as stream:
        for field,start,length in _fields(stream,file.stat().st_size):
            if field!=7: continue # ModelProto.graph
            stream.seek(start)
            for graph_field,node_start,node_length in _fields(stream,start+length):
                if graph_field!=1: continue # GraphProto.node
                stream.seek(node_start);name=''
                for node_field,text_start,text_length in _fields(stream,node_start+node_length):
                    if node_field==3: # NodeProto.name
                        if text_length>4096: raise ValueError('ONNX node name too long')
                        stream.seek(text_start);name=stream.read(text_length).decode('utf8')
                names.append(name)
                if len(names)==8: break
            break
    if not names: raise ValueError('ONNX recognition graph has no nodes')
    embedded=any(n.startswith(('Sub','_minus')) for n in names) and any(n.startswith(('Mul','_mul')) for n in names)
    return {'mean':0.0 if embedded else 127.5,'std':1.0 if embedded else 127.5,'embedded':embedded}
