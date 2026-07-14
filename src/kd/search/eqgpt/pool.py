
from __future__ import annotations

from kd.search.eqgpt.vocab import E_ID, PLUS_ID


def dedup_sentence(sentence: list[int]) -> list[int]:
    body = list(sentence)
    if body and body[-1] == E_ID:
        body.pop(-1)

    slices: list[list[int]] = []
    current: list[int] = []
    for word in body:
        if word != PLUS_ID:
            current.append(word)
        else:
            slices.append(current)
            current = []
    slices.append(current)

    seen_keys: list[list[int]] = []
    unique_slices: list[list[int]] = []
    for sl in slices:
        key = sorted(sl[::2])
        if key in seen_keys:
            continue
        seen_keys.append(key)
        unique_slices.append(sl)

    concise: list[int] = []
    for sl in unique_slices:
        concise.extend(sl)
        concise.append(PLUS_ID)
    concise.pop(-1)
    concise.append(E_ID)
    return concise


def merge_top_k(
    pool_rewards: list[float],
    pool_sentences: list[list[int]],
    new_rewards: list[float],
    new_sentences: list[list[int]],
    k: int,
) -> tuple[list[float], list[list[int]]]:
    paired_new = sorted(
        zip(new_rewards, new_sentences, strict=True), key=lambda p: -p[0]
    )

    if not pool_rewards:
        rewards: list[float] = []
        sentences: list[list[int]] = []
        for reward, sentence in paired_new:
            if reward in rewards:
                continue
            rewards.append(reward)
            sentences.append(sentence)
            if len(rewards) == k:
                break
        return rewards, sentences

    rewards = list(pool_rewards)
    sentences = list(pool_sentences)




    deduped_new: list[tuple[float, list[int]]] = []
    seen_new: list[float] = []
    for reward, sentence in paired_new:
        if reward in seen_new:
            continue
        seen_new.append(reward)
        deduped_new.append((reward, sentence))
        if len(deduped_new) == k:
            break

    for reward, sentence in deduped_new:
        if reward in rewards:
            continue
        insert_idx = next(
            (idx for idx, existing in enumerate(rewards) if reward > existing), None
        )
        if insert_idx is None:
            continue
        rewards.insert(insert_idx, reward)
        sentences.insert(insert_idx, sentence)
        rewards.pop(-1)
        sentences.pop(-1)

    return rewards, sentences
