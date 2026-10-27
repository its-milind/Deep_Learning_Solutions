def gradient_accumulator(
    theta: float, eta: float, microbatches: list[tuple[int, float]]
) -> tuple[float, float]:
    # TODO: weight by example count, divide exactly once.
    n_j_sum  = 0
    total = 0
    for n_j,g_j in microbatches:
        total += n_j*g_j
        n_j_sum += n_j
    g = total/n_j_sum
    theta_prime = theta - eta*g
    return (g, theta_prime)
