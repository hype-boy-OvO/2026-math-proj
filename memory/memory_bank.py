import numpy as np

class MemoryBank:
    def __init__(self):
        self.memory = []

    def fourier_series(self, f, num_bases=350, num_peak=550, N=3072):
        f = np.array(f)
        f = f.flatten()
        f = np.fft.rfft(f)
        an = f.real * (2.0 / N)
        bn = -f.imag * (2.0 / N)

        an[0] = an[0] / 2.0
        if N % 2 == 0:
            an[-1] = an[-1] / 2.0

        low_an = an[:num_bases]
        low_bn = bn[:num_bases]

        remaining_an = an[num_bases:]
        remaining_bn = bn[num_bases:]
        magnitudes = remaining_an**2 + remaining_bn**2
        peak_indices = np.argsort(magnitudes)[-num_peak:]
        high_an = remaining_an[peak_indices]
        high_bn = remaining_bn[peak_indices]
        high_idx = peak_indices + num_bases

        return{
            "low_an": low_an,
            "low_bn": low_bn,
            "high_an": high_an,
            "high_bn": high_bn,
            "high_idx": high_idx
        }

    def fourier_is_close(self, value):

        def cal_relative_error(a, b,sim_threshold=0.14):
            v = np.abs(a-b) / (np.abs(a) + np.abs(b) + 1e-8)

            if v <= (sim_threshold*2.0):
                return True
            else:
                return False
        

        low_count, high_count = 0, 0
        result = []
        
        if not self.memory:
            return []
        
        for i, mem in enumerate(self.memory):
            mem = mem["fourier"]

            for k in range(len(mem["low_an"])):
                if cal_relative_error(mem["low_an"][k], value["low_an"][k]) and cal_relative_error(mem["low_bn"][k], value["low_bn"][k]):
                    low_count += 1

            for k in range(len(mem["high_an"])):
                if cal_relative_error(mem["high_idx"][k], value["high_idx"][k],sim_threshold=0.075):
                    if (cal_relative_error(mem["high_an"][k], value["high_an"][k]) and cal_relative_error(mem["high_bn"][k], value["high_bn"][k])):
                        high_count += 1
            
            idx = i

            if (low_count*(85/350) + high_count*(15/550)) >= 20:
                result.append(idx)

            low_count, high_count = 0, 0

        return result
            

    def add_memory(self, fourier, text):
        self.memory.append({"fourier": fourier, "text": text})
