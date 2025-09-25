import json
import torch
from functools import partial
from gliner.decoding.utils import has_overlapping, has_overlapping_nested
from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
from sklearn.model_selection import KFold
from tqdm import tqdm
import os
import glob

def gliner_format(example):
    annotations = [(span["start"], span["end"] - 1, span["tag"]) for span in example["token_spans"]]
    return {"tokenized_text": example["tokens"], "ner": annotations}

def greedy_search(spans, flat_ner=True, multi_label=False):
    if flat_ner:
        has_ov = partial(has_overlapping, multi_label=multi_label)
    else:
        has_ov = partial(has_overlapping_nested, multi_label=multi_label)

    new_list = []
    span_prob = sorted(spans, key=lambda x: -x[-1])

    for i in range(len(spans)):
        b = span_prob[i]
        flag = False
        for new in new_list:
            if has_ov(b[:-1], new):
                flag = True
                break
        if not flag:
            new_list.append(b)

    new_list = sorted(new_list, key=lambda x: x[0])
    return new_list

def get_indices_above_threshold(scores, threshold):
    scores = torch.sigmoid(scores)
    return [k.tolist() for k in torch.where(scores > threshold)]

def calculate_span_score(start_idx, end_idx, scores_inside_i, start_i, end_i, id_to_classes, threshold):
    span_i = []
    for st, cls_st in zip(*start_idx):
        for ed, cls_ed in zip(*end_idx):
            if ed >= st and cls_st == cls_ed:
                ins = scores_inside_i[st:ed + 1, cls_st]
                if (ins < threshold).any():
                    continue
                # Get the start and end scores for this span
                start_score = start_i[st, cls_st]
                end_score = end_i[ed, cls_st]
                # Concatenate the inside scores with start and end scores
                combined = torch.cat([ins, start_score.unsqueeze(0), end_score.unsqueeze(0)])
                # The span score is the minimum value among these scores
                spn_score = combined.min().item()
                span_i.append((st, ed, id_to_classes[cls_st + 1], None, spn_score))
    return span_i

def main():
    for fold in range(5):
        model_path = f"/vol/tmp/goldejon/multilingual_ner/gliner_logs/fold_{fold+1}/checkpoint-50000/"
        model = GLiNER.from_pretrained(model_path)
        model.eval()
        model.to("cuda")

        data = []
        for lang_path in glob.glob(f"/vol/tmp2/goldejon/multilingual_ner/data/singlelabel/finerweb_merged_jsonl/*.jsonl"):
            with open(lang_path, 'r') as f:
                lang_data = [
                    {'original': json.loads(item), 'gliner_format': gliner_format(json.loads(item))} for item in f.readlines()]
            data.extend(lang_data)

        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        print('Dataset size:', len(data))
        splits = list(kf.split(data))
        print(f'Dataset is split into {k_folds} folds...')

        for fold in range(k_folds):
            test_indices = splits[fold][1]
            test_data = [data[i] for i in test_indices]

            collator = DataCollator(
                model.config,
                data_processor=model.data_processor,
                return_tokens=True,
                return_entities=True,
                return_id_to_classes=True,
                prepare_labels=False,
            )

            data_loader = torch.utils.data.DataLoader(
                test_data, batch_size=1, shuffle=False, collate_fn=collator
            )

            annotations = []
            for batch in tqdm(data_loader):
                original_input = batch.pop('original')
                # Move the batch to the appropriate device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(model.device)

                # Perform predictions
                with torch.no_grad():
                    model_output = model(**batch)[0]

                if not isinstance(model_output, torch.Tensor):
                    model_output = torch.from_numpy(model_output)

                model_output = model_output.permute(3, 0, 1, 2)
                scores_start, scores_end, scores_inside = model_output
                threshold = 0.4
                id_to_classes = batch["id_to_classes"]
                for batch_idx, _ in enumerate(batch["tokens"]):
                    gold_spans = batch["entities"][batch_idx]
                    id_to_class_i = id_to_classes[batch_idx] if isinstance(id_to_classes, list) else id_to_classes
                    class_to_id_i = {v: k for k, v in id_to_class_i.items()}
                    start_tuple = [[], []]
                    end_tuple = [[], []]
                    for span in gold_spans:
                        start_tuple[0].append(span[0])
                        start_tuple[1].append(class_to_id_i[span[2]] - 1)
                        end_tuple[0].append(span[1])
                        end_tuple[1].append(class_to_id_i[span[2]] - 1)

                    gold_spans = calculate_span_score(
                        start_tuple,
                        end_tuple,
                        torch.sigmoid(scores_inside[batch_idx]),
                        torch.sigmoid(scores_start[batch_idx]),
                        torch.sigmoid(scores_end[batch_idx]),
                        id_to_class_i,
                        threshold
                    )
                    gold_spans = greedy_search(gold_spans, True, False)

                    pred_spans = calculate_span_score(
                        get_indices_above_threshold(scores_start[batch_idx], threshold),
                        get_indices_above_threshold(scores_end[batch_idx], threshold),
                        torch.sigmoid(scores_inside[batch_idx]),
                        torch.sigmoid(scores_start[batch_idx]),
                        torch.sigmoid(scores_end[batch_idx]),
                        id_to_class_i,
                        threshold
                    )
                    pred_spans = greedy_search(pred_spans, True, False)
                    annotations.append({**original_input[0], "gold_spans_with_confidence": gold_spans, "pred_spans_with_confidence": pred_spans})

            os.makedirs(f"/vol/tmp/goldejon/multilingual_ner/data/confidence_annotations", exist_ok=True)
            with open(f"/vol/tmp/goldejon/multilingual_ner/data/confidence_annotations/fold_{fold+1}.json", "w") as f:
                json.dump(annotations, f)

if __name__ == "__main__":
    main()