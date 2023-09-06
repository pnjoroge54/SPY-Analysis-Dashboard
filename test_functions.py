def weighted_harmonic_mean(array, weights):
    if len(array) != len(weights):
        return "Weights missing from array item(s)"
    else:
        d = sum([w / x for x, w in zip(array, weights)])
        n = sum(weights)
        return n / d
        

def test_weighted_harmonic_mean(array, weights):
    assert round(weighted_harmonic_mean([2.5, 3, 10], [1, 1, 1]), 1) == 3.6