import torch

def greedy_decode(
    model,
    x,
    normalizer,
    bos_id,
    eos_id,
    max_length,
):

    model.eval()

    x = normalizer(x)

    batch_size = x.size(0)

    # Start decoder with BOS
    decoder_input = torch.full(
        (batch_size, 1),
        bos_id,
        dtype=torch.long,
        device=x.device,
    )

    generated = []

    with torch.no_grad():

        for _ in range(max_length):

            logits = model(
                x,
                decoder_input,
            )

            # Only need the newest timestep
            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated.append(
                next_token
            )

            decoder_input = torch.cat(
                [
                    decoder_input,
                    next_token,
                ],
                dim=1,
            )

            # Stop if every sequence generated EOS
            if torch.all(
                next_token.squeeze(1)
                == eos_id
            ):

                break

    if len(generated) == 0:

        return torch.empty(
            batch_size,
            0,
            dtype=torch.long,
            device=x.device,
        )

    return torch.cat(
        generated,
        dim=1,
    )