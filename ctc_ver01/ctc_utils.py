import torch
import numpy as np

# Inference
def beam_search(logits, beam_width, blank_id, greedy=True, sample=True):
    # Preprocess
    log_prob = torch.log_softmax(logits, dim=2).squeeze().cpu().numpy()

    # At t = 0 => Take highest probabilities
    best_beams = np.argsort(log_prob[0])[-beam_width:]
    beams = [([int(i), ], log_prob[0, i]) for i in best_beams]

    for t in range(1, log_prob.shape[0]):
        new_beams = []
        for b in beams:
            beam_score = b[1]
            beam = b[0]
            best_beams = np.argsort(log_prob[t])[-beam_width:]
            new_beams += [(beam + [int(i)], beam_score + log_prob[t, i]) for i in best_beams]

            # Case 1: current beam not ends with blank but token X, copy beam can be X + X or X + blank => consider combine effect
            if beam[-1] != blank_id:
                lp_copy = log_prob[t, blank_id] + log_prob[t, beam[-1]]
                lp_copy_score = lp_copy + beam_score
                new_beams += [(beam + [blank_id], lp_copy_score), (beam + [int(beam[-1])], lp_copy_score)]

            # Case 2: current beam ends with blank, copy beam can only be the next blank => no combine effect

        # Choose next best beams
        scores = [b[1] for b in new_beams]
        best_beams = np.argsort(scores)[-beam_width:]
        beams = [new_beams[i] for i in best_beams]

    if not sample:
        return beams

    scores = torch.stack([
        b[1] if torch.is_tensor(b[1]) else torch.tensor(b[1])
        for b in beams
    ])

    prob = torch.softmax(scores, dim=0)
    prob_np = prob.cpu().numpy()

    if greedy:
        sampled_beam_idx = torch.argmax(scores).item()
    else:
        sampled_beam_idx = np.random.choice(
            len(scores),
            size=1,
            p=prob_np
        )[0]

    return beams[sampled_beam_idx], prob_np[sampled_beam_idx]

def collpase(tokens, blank_id):
    collpased = []
    for i in range(0, len(tokens)):
        if len(collpased) == 0:
            if tokens[i] != blank_id:
                collpased.append(tokens[i])
        else:
            if tokens[i] != blank_id and (tokens[i] != collpased[-1] or tokens[i-1] == blank_id):
                collpased.append(tokens[i])
            
    return collpased