import argparse
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();Path(a.output).parent.mkdir(parents=True,exist_ok=True);quantize_dynamic(a.input,a.output,weight_type=QuantType.QInt8);print(a.output)
if __name__=='__main__':main()
