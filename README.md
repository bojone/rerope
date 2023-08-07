# Rectified Rotary Position Embeddings (ReRoPE)

Using ReRoPE, we can more effectively extend the context length of LLM without the need for fine-tuning.

## Blog

- https://kexue.fm/archives/9706
- https://kexue.fm/archives/9708

## Idea

$$\begin{pmatrix}0 & \\ 
1 & 0 & \\ 
2 & 1 & 0 &\\ 
3 & 2 & 1 & 0 & \\ 
\ddots & 3 & 2 & 1 & 0 & \\ 
\ddots & \ddots & 3 & 2 & 1 & 0 & \\ 
\ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots \\ 
\tiny{L - 2} & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots \\ 
\tiny{L - 1} & \tiny{L - 2} & \ddots & \ddots & \ddots & 3 & 2 & 1 & 0 & \\ 
\end{pmatrix}\quad\to\quad\begin{pmatrix} 
\color{red}{0} & \\ 
\color{red}{1} & \color{red}{0} & \\ 
\color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{red}{\tiny{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{w} & \color{red}{\tiny{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{w} & \color{green}{w} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{\ddots} & \color{green}{w} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \\ 
\color{green}{w} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{w} & \color{green}{w} & \color{red}{\tiny{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\end{pmatrix}$$

## Cite

## 引用

```
@misc{rerope2023,
  title={Rectified Rotary Position Embeddings},
  author={Jianlin Su},
  year={2023},
  howpublished={\url{https://github.com/bojone/rerope}},
}
```
