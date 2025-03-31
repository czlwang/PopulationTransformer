from torch.nn.utils.rnn import pad_sequence
import torch

def make_pad_mask(batched_input, lengths):
    pad_mask = torch.ones(batched_input.shape[:-1]) #[batch, len]

    for i in range(pad_mask.shape[0]):
        pad_mask[i,lengths[i]:] = 0

    pad_mask = ~pad_mask.bool() 
    return pad_mask

def spec_collator(batch):
    input_specs = [b["masked_input"] for b in batch]
    mask_labels = [b["mask_label"] for b in batch]
    targets = [b["target"] for b in batch]
    lengths = [b["length"] for b in batch]
    wavs = [b["wav"] for b in batch]

    batched_input = pad_sequence(input_specs, batch_first=True)
    batched_target = pad_sequence(targets, batch_first=True)
    batched_mask_label = pad_sequence(mask_labels, batch_first=True)

    attn_mask = make_pad_mask(batched_input, lengths)

    batch = {"attn_mask": attn_mask,
             "masked_input": batched_input,
             "target": batched_target,
             "mask_label": batched_mask_label,
             "wavs": wavs}
    return batch

def wav_collator(batch):
    wavs = [torch.Tensor(b["input"]).unsqueeze(0) for b in batch]
    wavs = pad_sequence(wavs, batch_first=True)
    return {"input":wavs,
           }

def baseline_wav_collator(batch):
    labels = [b["label"] for b in batch]
    wavs = [torch.Tensor(b["input"]) for b in batch]
    wavs = pad_sequence(wavs, batch_first=True)

    lengths = [b["length"] for b in batch]

    return {"input":wavs,
            "labels":labels,
           }

def finetune_collator(batch):
    specs = [b["input"] for b in batch]
    specs = pad_sequence(specs, batch_first=True)
    labels = [b["label"] for b in batch]
    wavs = [b["wav"] for b in batch]

    lengths = [b["length"] for b in batch]
    pad_mask = make_pad_mask(specs, lengths)

    return {"input":specs,
            "labels":labels,
            "wavs": wavs,
            "pad_mask": pad_mask}

def full_brain_feature_extracter_collator(batch):
    specs = [b["input"] for b in batch]
    #specs = pad_sequence(specs, batch_first=True)
    labels = [b["label"] for b in batch]
    wavs = [b["wav"] for b in batch]

    lengths = [b["length"] for b in batch]

    return {"input":specs,
            "labels":labels,
            "wavs": wavs}

def multi_subj_feature_extracter_collator(batch):
    specs = [b["input"] for b in batch]
    specs = pad_sequence(specs, batch_first=True)
    labels = [b["label"] for b in batch]
    wavs = [b["wav"] for b in batch]

    lengths = [b["length"] for b in batch]

    return {"input":specs,
            "labels":labels,
            "wavs": wavs}

def pt_feature_extract_collator(batch):
    specs = [b["input"] for b in batch]
    specs = pad_sequence(specs, batch_first=True)

    labels = [b["label"] for b in batch]
    wavs = [b["wav"] for b in batch]
    positions = [b["position"] for b in batch]

    lengths = [b["length"] for b in batch]
    attn_mask = make_pad_mask(specs, lengths)

    return {"input":specs,
            "labels":labels,
            "attn_mask":attn_mask,
            "position":positions,
           }

def pretrain_collator(batch):
    specs = [b["input"] for b in batch]
    specs = pad_sequence(specs, batch_first=True)

    targets = [b["target"] for b in batch]
    targets = pad_sequence(targets, batch_first=True)

    mask_labels = [b["mask_label"] for b in batch]
    #assert targets.shape==specs.shape

    labels = [b["label"] for b in batch]
    wavs = [b["wav"] for b in batch]
    batched_mask_label = pad_sequence(mask_labels, batch_first=True)

    lengths = [b["length"] for b in batch]
    attn_mask = make_pad_mask(specs, lengths)
    positions = [b["position"] for b in batch]

    return {"input":specs,
            "labels":labels,
            "attn_mask":attn_mask,
            "target": targets,
            "mask_label": batched_mask_label,
            "position":positions,
           }

def pt_feature_extract_coords_collator(batch):
    specs = [b["input"] for b in batch]
    specs = pad_sequence(specs, batch_first=True)

    labels = [b["label"] for b in batch]

    lengths = [b["length"] for b in batch]
    attn_mask = make_pad_mask(specs, lengths)

    coords = [b["coords"] for b in batch]
    coords = pad_sequence(coords, batch_first=True)

    seq_id = [b["seq_id"] for b in batch]
    batched_seq_id = pad_sequence(seq_id, batch_first=True)

    return {"input":specs,
            "labels":labels,
            "attn_mask":attn_mask,
            "coords":coords,
            "seq_id": batched_seq_id
           }

def twins_pretrain_collator(batch):
    specs_1 = [b["input_1"] for b in batch]
    specs_1 = pad_sequence(specs_1, batch_first=True)

    specs_2 = [b["input_2"] for b in batch]
    specs_2 = pad_sequence(specs_2, batch_first=True)

    targets_1 = [b["targets_1"] for b in batch]
    targets_1 = pad_sequence(targets_1, batch_first=True)

    targets_2 = [b["targets_2"] for b in batch]
    targets_2 = pad_sequence(targets_2, batch_first=True)

    mask_labels_1 = [b["mask_labels_1"] for b in batch]
    mask_labels_1 = pad_sequence(mask_labels_1, batch_first=True)

    mask_labels_2 = [b["mask_labels_2"] for b in batch]
    mask_labels_2 = pad_sequence(mask_labels_2, batch_first=True)

    labels = [b["label"] for b in batch]
    wavs = [b["wav"] for b in batch]

    lengths_1 = [b["length_1"] for b in batch]
    attn_mask_1 = make_pad_mask(specs_1, lengths_1)

    lengths_2 = [b["length_2"] for b in batch]
    attn_mask_2 = make_pad_mask(specs_2, lengths_2)

    coords_1 = [b["coords_1"] for b in batch]
    coords_1 = pad_sequence(coords_1, batch_first=True)

    coords_2 = [b["coords_2"] for b in batch]
    coords_2 = pad_sequence(coords_2, batch_first=True)

    return {"input_1":specs_1,
            "input_2":specs_2,
            "labels":labels,
            "attn_mask_1":attn_mask_1,
            "attn_mask_2":attn_mask_2,
            "targets_1": targets_1,
            "targets_2": targets_2,
            "mask_labels_1": mask_labels_1,
            "mask_labels_2": mask_labels_2,
            "coords_1":coords_1,
            "coords_2":coords_2,
           }

def nsp_time_pretrain_collator(batch):
    specs = [b["input"] for b in batch]
    specs = pad_sequence(specs, batch_first=True)

    targets = [b["target"] for b in batch]
    targets = pad_sequence(targets, batch_first=True)

    mask_labels = [b["mask_label"] for b in batch]
    #assert targets.shape==specs.shape

    labels = [b["label"] for b in batch]
    wavs = [b["wav"] for b in batch]
    batched_mask_label = pad_sequence(mask_labels, batch_first=True)

    #replace_labels = [b["replace_label"] for b in batch]
    #batched_replace_label = pad_sequence(replace_labels, batch_first=True)

    lengths = [b["length"] for b in batch]
    attn_mask = make_pad_mask(specs, lengths)

    coords = [b["coords"] for b in batch]
    coords = pad_sequence(coords, batch_first=True)

    seq_id = [b["seq_id"] for b in batch]
    batched_seq_id = pad_sequence(seq_id, batch_first=True)

    time_id = [b["time_id"] for b in batch]
    batched_time_id = pad_sequence(time_id, batch_first=True)


    return {"input":specs,
            "labels":labels,
            "attn_mask":attn_mask,
            "target": targets,
            "mask_label": batched_mask_label,
            "coords":coords,
            "seq_id": batched_seq_id,
            "time_id": batched_seq_id
           }

def connectivity_collator(batch):
    specs = [b["input"] for b in batch]
    specs = pad_sequence(specs, batch_first=True)

    targets = [b["target"] for b in batch]
    targets = pad_sequence(targets, batch_first=True)

    alternatives = [b["alternative"] for b in batch]
    alternatives = pad_sequence(alternatives, batch_first=True)

    mask_labels = [b["mask_label"] for b in batch]
    #assert targets.shape==specs.shape

    labels = [b["label"] for b in batch]
    wavs = [b["wav"] for b in batch]
    batched_mask_label = pad_sequence(mask_labels, batch_first=True)

    replace_labels = [b["replace_label"] for b in batch]
    batched_replace_label = pad_sequence(replace_labels, batch_first=True)

    lengths = [b["length"] for b in batch]
    attn_mask = make_pad_mask(specs, lengths)

    coords = [b["coords"] for b in batch]
    coords = pad_sequence(coords, batch_first=True)

    seq_id = [b["seq_id"] for b in batch]
    batched_seq_id = pad_sequence(seq_id, batch_first=True)

    return {"input":specs,
        "labels":labels,
        "attn_mask":attn_mask,
        "target": targets,
        "replace_label": batched_replace_label,
        "coords":coords,
        "seq_id": batched_seq_id,
        "alternative": alternatives,
       }

def nsp_replace_only_pretrain_collator(batch):
    specs = [b["input"] for b in batch]
    specs = pad_sequence(specs, batch_first=True)

    targets = [b["target"] for b in batch]
    targets = pad_sequence(targets, batch_first=True)

    #assert targets.shape==specs.shape

    labels = [b["label"] for b in batch]
    wavs = [b["wav"] for b in batch]

    replace_labels = [b["replace_label"] for b in batch]
    batched_replace_label = pad_sequence(replace_labels, batch_first=True)

    lengths = [b["length"] for b in batch]
    attn_mask = make_pad_mask(specs, lengths)

    coords = [b["coords"] for b in batch]
    coords = pad_sequence(coords, batch_first=True)

    seq_id = [b["seq_id"] for b in batch]
    batched_seq_id = pad_sequence(seq_id, batch_first=True)

    return {"input":specs,
            "labels":labels,
            "attn_mask":attn_mask,
            "target": targets,
            "replace_label": batched_replace_label,
            "coords":coords,
            "seq_id": batched_seq_id,
           }

def nsp_replace_pretrain_collator(batch):
    specs = [b["input"] for b in batch]
    specs = pad_sequence(specs, batch_first=True)

    targets = [b["target"] for b in batch]
    targets = pad_sequence(targets, batch_first=True)

    mask_labels = [b["mask_label"] for b in batch]
    #assert targets.shape==specs.shape

    labels = [b["label"] for b in batch]
    wavs = [b["wav"] for b in batch]
    batched_mask_label = pad_sequence(mask_labels, batch_first=True)

    replace_labels = [b["replace_label"] for b in batch]
    batched_replace_label = pad_sequence(replace_labels, batch_first=True)

    lengths = [b["length"] for b in batch]
    attn_mask = make_pad_mask(specs, lengths)

    coords = [b["coords"] for b in batch]
    coords = pad_sequence(coords, batch_first=True)

    seq_id = [b["seq_id"] for b in batch]
    batched_seq_id = pad_sequence(seq_id, batch_first=True)

    return {"input":specs,
            "labels":labels,
            "attn_mask":attn_mask,
            "target": targets,
            "mask_label": batched_mask_label,
            "replace_label": batched_replace_label,
            "coords":coords,
            "seq_id": batched_seq_id,
           }


def nsp_pretrain_collator(batch):
    specs = [b["input"] for b in batch]
    specs = pad_sequence(specs, batch_first=True)

    targets = [b["target"] for b in batch]
    targets = pad_sequence(targets, batch_first=True)

    mask_labels = [b["mask_label"] for b in batch]
    #assert targets.shape==specs.shape

    labels = [b["label"] for b in batch]
    wavs = [b["wav"] for b in batch]
    batched_mask_label = pad_sequence(mask_labels, batch_first=True)

    lengths = [b["length"] for b in batch]
    attn_mask = make_pad_mask(specs, lengths)

    coords = [b["coords"] for b in batch]
    coords = pad_sequence(coords, batch_first=True)

    seq_id = [b["seq_id"] for b in batch]
    batched_seq_id = pad_sequence(seq_id, batch_first=True)


    return {"input":specs,
            "labels":labels,
            "attn_mask":attn_mask,
            "target": targets,
            "mask_label": batched_mask_label,
            "coords":coords,
            "seq_id": batched_seq_id,
           }

def feature_extracter_collator(batch):
    specs = [b["input"] for b in batch]
    specs = pad_sequence(specs, batch_first=True)
    labels = [b["label"] for b in batch]
    wavs = [b["wav"] for b in batch]

    lengths = [b["length"] for b in batch]

    return {"input":specs,
            "labels":labels,
            "wavs": wavs}
