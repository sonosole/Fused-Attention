# FusedAttention: fusing attention matrix into attention vector

> *If some formula are not displayed clearly, chose the pdf format* **README.pdf** *in this repository.*

## Self-Attention

Let $X \in ℝ^{h \times L}$ denotes a sequence of $L$ feature vectors of dimensions $h$. Formally, $X$ is projected by three matrices $W_{Q} \in ℝ^{u \times h}$, $W_{K} \in ℝ^{u \times h}$ and $W_{V} \in ℝ^{d \times h}$ to corresponding representations $Q$, $K$ and $V$. The output for all positions is computed as follows

$$
\begin{align*}
Q &= W_{Q} * X, \\
K &= W_{K} * X, \\
V &= W_{V} * X, \\
α &= {\rm softmax}(K^{T} Q  / \sqrt{d}), \\
Y &= V * α.
\end{align*}
$$

Note that the softmax funtion is appied column-wise. The  $Q$, $K$, $V$ and $α$ are referred to as the queries, keys, values, and attention matrix respectively, following the common terminology.

## Fused-Attention

In self-attention, the output $Y$ at position $t$ is computed as a weighted average of the feature representations of all positions with a weight proportional to a similarity score between the representations.

$$
\begin{align*}
Q_{t} &= W_{Q} X_{t}, \\
K &= W_{K} X, \\
V &= W_{V} X, \\
α_{t} &= {\rm softmax}(K^{T} Q_{t}  / \sqrt{d}), \\
Y_{t} &= V * α_{t}.
\end{align*}
$$

In short, self-attention maps $L$ inputs to $L$ outputs. In fused-attention, we rewrite the formula as

$$
\begin{align*}
Q &= f_Q (X), \\
K &= f_K (X), \\
V &= f_V (X), \\
α &= fuse \left( norm( K^{T} Q / \sqrt{d} ) \right), \\
Y &= g_{V}(V * α),
\end{align*}
$$

where $f_Q$, $f_K$ and $f_V$ are any functions that are legal to input $X$, $norm$ is a normalization function default to softmax appied column-wise, $fuse$ is an aggregation function default to mean appied row-wise, $g_V$ is a transform function related to $V$, thus the attention variable $α \in ℝ^{L \times 1}$ transforms values $V$ from a matrix $\in ℝ^{d \times L}$ to a vector $\in ℝ^{d \times 1}$. In brief, fused-attention maps $L$ inputs to $1$ output. There are two forms of $X$：

+ The first case is that $X$ is a tensor of any size. Ignoring the batch dimension, suppose the size of $X$ is $(C,W_1,W_2,...,W_n)$, where $C$ is the number of channels, $W_n$ is the spatial width at the $n$-th spatial dimention. In this case, take $f_V$ for example, it first transforms $X$ into $V'$ with size $(C',W_1',W_2',...,W_n')$, then transforms $V'$ into $V$ with size $(d,L)$, where $d = C' \prod\nolimits_{i=1}^{n} ΔW_i'$, of which $ΔW_i'$ is the window size at the $i$-th spatial dimention, $L$ is the total number of patches. $f_Q$ and $f_K$ are similar to $f_V$. If the number of dimentions of $Y$ is required to be the same with $X$, then $g_{V}$ transforms $V*α$ into $Y$ with size $(C', ΔW_1',ΔW_2',...,ΔW_n')$, otherwise $g_{V}$ does nothing.
+ The second case is that $X$ is a collection of $L$ tensors of the same size. Also ignoring the batch dimension, $X_j$ is the $j$-th element of $X$, suppose the size of $X_j$ is $(C,W_1,W_2,...,W_n)$, where $C$ is the number of channels, $W_n$ is the spatial width at the $n$-th spatial dimention. In this case, take $f_V$ for example, it first transforms $X_j$ into $V_j'$ with size $(C',W_1',W_2',...,W_n')$, then reshape $V_j'$ to $V_j$ of size $(C' \prod\nolimits_{i=1}^{n} W_i', 1)$, then concatenate $V_j, j=1,2,...,L$ into $V$ with size $(C' \prod\nolimits_{i=1}^{n} W_i', L)$. $f_Q$ and $f_K$ are similar to $f_V$. As to $g_{V}$, it transforms $V*α$ into $Y$ with size $(C', W_1',W_2',...,W_n')$.
