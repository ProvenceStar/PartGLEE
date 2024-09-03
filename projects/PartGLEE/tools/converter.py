import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glee_weight_path', default='projects/PartGLEE/checkpoint/GLEE_Lite_scaleup.pth')
    parser.add_argument('--output_path', default='projects/PartGLEE/checkpoint/PartGLEE_converted_from_GLEE_RN50.pth')
    args = parser.parse_args()
    weights = torch.load(args.glee_weight_path, map_location='cpu')
    converted_weights = {}
    for key in weights.keys():
        converted_key = key.replace('glee', 'partglee')
        if 'predictor' in key:
            converted_weights[converted_key.replace('predictor', 'object_predictor')] = weights[key]
            converted_weights[converted_key.replace('predictor', 'part_predictor')] = weights[key]
        else:    
            converted_weights[converted_key] = weights[key]
    torch.save(converted_weights, args.output_path)