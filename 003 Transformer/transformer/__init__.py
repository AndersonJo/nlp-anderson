from transformer.models import Transformer


def get_transformer(opt) -> Transformer:
    # Encoder hyper-parameters
    model = Transformer(embed_dim=opt.embed_dim,
                        src_vocab_size=opt.src_vocab_size,
                        trg_vocab_size=opt.trg_vocab_size,
                        src_pad_idx=opt.src_pad_idx,
                        trg_pad_idx=opt.trg_pad_idx,
                        n_head=opt.n_head)
    model = model.to(opt.device)
    return model
