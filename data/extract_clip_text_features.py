import os
import json
import pickle
import nltk
import torch
import numpy as np


from tqdm import tqdm
import run_on_video.clip as clip


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]
    
if __name__ == '__main__':
    vocab_size = 1111

    # anno_path = './anet/charades_sta_train_tvr_format.jsonl'
    anno_path = './anet/charades_sta_test_tvr_format.jsonl'
    annotations = load_jsonl(anno_path)

    model, preprocess = clip.load("ViT-B/32", jit=False)
    model.cuda().eval()
    for anno in tqdm(annotations, desc="Generating text features"):
        # import pdb;pdb.set_trace()
        query = anno['query']
        qid = anno['qid']
        text_tokens = clip.tokenize(query).cuda()

        context_length = model.context_length
        vocab_size = model.vocab_size
        # image = torch.rand([1,3,224,224]).cuda()
        with torch.no_grad():
            # image_feature, output = model.encode_image(image)
            text_feature = model.encode_text(text_tokens)
            last_hidden_state = text_feature['last_hidden_state'].squeeze(0).cpu().numpy()
            pooler_output = text_feature['pooler_output'].squeeze(0).cpu().numpy()

        save_npz_name = f"qid{qid}.npz"
        save_path = os.path.join('/mnt/workspace/workgroup/multimodal/moment_ret/charades/clip_text_feature/', save_npz_name)

        np.savez(save_path, last_hidden_state=last_hidden_state, pooler_output=pooler_output)
