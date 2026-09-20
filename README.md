# plaquette

> *plaquette*: in lattice gauge theory, the smallest object built from link variables that does not depend on your choice of gauge.

Measuring the gauge structure of a single attention head: conserved charges, flat directions, and what survives a change of basis.

---

## The idea

*"The lamp is left of the sofa."* Walk to the other side of the room and it is right of the sofa. Nothing moved. That sentence was never about the furniture. It was about where you were standing.

*"The lamp is two metres from the sofa"* holds wherever you stand.

This repository walks to the other side of the room.

## What that means

Inside an attention head, `W_Q` and `W_K` only ever appear multiplied together. So replace them (`@` is matrix multiplication in Python, `.T` is transpose):

```
W_Q  ->  W_Q @ M
W_K  ->  W_K @ inv(M).T
```

for any invertible `M`, and every output is identical. Bit for bit. The weights changed completely. The network did not.

> **A claim about a network only means something if it survives that transformation.**

## Why bother

Mechanistic interpretability opens up trained models and works out what they compute. It has real results: heads responsible for specific behaviours, named features you can turn up and down, a model trained only on Othello moves that carries a picture of the board.

Some of those claims are two metres. Some are left of the sofa. Both get published in the same tone.

## What is in here

**1. One attention head from scratch.** Twenty lines, no library shortcuts. Apply the transformation, confirm the attention pattern does not move. If it moves, stop and find out why.

**2. Something to learn.** `(a + b) mod 13`, all 169 pairs. Two lines of code, no downloads, nowhere for a data bug to hide. Hold out 30% to tell learning from memorising.

**3. A conservation law, broken.** `W_Q.T @ W_Q - W_K.T @ W_K` is exactly conserved under gradient flow. Real training is not gradient flow, so it breaks, and how it breaks depends on the optimiser:

| optimiser | prediction |
|---|---|
| plain SGD | stays near zero |
| SGD + weight decay | climbs steadily to 1 |
| Adam | large and erratic |

Weight decay is a gauge fixing term, so it drives the charge to zero. Adam scales each coordinate separately, which is not basis independent, so it wanders off the orbit. Predictions made before the plots exist.

**4. Count the flat directions.** A flat direction changes the weights and nothing else. The architecture predicts `d_head` squared of them. Stack the gradients over all 169 inputs, take the SVD, count the singular values near zero.

Predict a number, measure the number, see whether they agree. This is an instrument, and step 4 is where you learn whether it reads correctly.

## References

- François & Ravera (2026), *Toward Manifest Relationality in Transformers via Symmetry Reduction*, [arXiv:2602.18948](https://arxiv.org/abs/2602.18948). The formal version of this framing.
- Kunin et al. (2020), *Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics*, [arXiv:2012.04728](https://arxiv.org/abs/2012.04728). Where the conserved quantities come from.
- Zhao et al. (2022), *Symmetries, Flat Minima and the Conserved Quantities of Gradient Flow*, [arXiv:2210.17216](https://arxiv.org/abs/2210.17216).
- Posfai et al. (2025), *Symmetry, gauge freedoms, and the interpretability of sequence-function relationships*, Phys. Rev. Research **7**, 023005. The same argument, in genomics.
- Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits*, [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html). Arrives at the invariant objects empirically.