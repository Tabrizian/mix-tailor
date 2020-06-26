class Quantizer:

    def __init__(self, bits):
        super().__init__()
        self.bits = bits

    def quantize(self, grad):
        # returns a quantized vector, it is a bytestream in zero and one
        # it will be deqauntized in 8 bytes buckets and contacetaned
        return grad

    def step(self):
        pass


class Exponential(Quantizer):

    def __init__(self, bits, norm, p):
        # p is the exp factor
        super().__init__()
        self.bits = bits
        self.norm = norm

        num_levels = 2 << bits - 1

        self.levels = sum([[-p**j for j in range(num_levels >> 1)],
                           [p**j for j in reversed(range(num_levels >> 1))]],
                          [])

    def quantize(self, grad):
        pass
