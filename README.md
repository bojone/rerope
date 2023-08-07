# Rectified Rotary Position Embeddings (ReRoPE)

Using ReRoPE, we can more effectively extend the context length of LLM without the need for fine-tuning.

## Blog

- https://kexue.fm/archives/9706
- https://kexue.fm/archives/9708

## Idea

<img src="https://raw.githubusercontent.com/bojone/rerope/main/idea.png" width=750>

## Usage

Dependency: `transformers 4.31.0`

Run: `python test.py`

From [here](https://github.com/bojone/rerope/commit/2cbd019fcafec5ebe2bd9a2aec139c13ee8a67ae#diff-95a34212b33ed7212a3e43a00a8c7461378b45c35cce0e093d7f6ff068263670) and [here](https://github.com/bojone/rerope/commit/2cbd019fcafec5ebe2bd9a2aec139c13ee8a67ae#diff-f2d0565fa79ad02ed55bcaaea148d153d231641920488e51ae6b51f3e30cb464), we can see what modifications ReRoPE/Leaky ReRoPE has made compared to the original llama implementation.

## Cite

```
@misc{rerope2023,
  title={Rectified Rotary Position Embeddings},
  author={Jianlin Su},
  year={2023},
  howpublished={\url{https://github.com/bojone/rerope}},
}
```

## Communication
QQ discussion group: 67729435, for WeChat group, please add the robot WeChat ID spaces_ac_cn
