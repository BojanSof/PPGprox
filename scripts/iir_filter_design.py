import numpy as np
import scipy.signal as sig


def cmsis_dsp_coeffs(sos):
    """Generate CMSIS-DSP coeffs from SciPy sos"""
    # SciPy order of coeffs in biquad: b0, b1, b2, a0, a1, a2
    # CMSIS-DSP order of coeffs in biquad: b0 b1 b2 a1 a2
    # CMSIS-DSP assumes a0 = 1, so normalize the coeffs with a0
    # It is also needed to invert a1 and a2 from SciPy for CMSIS-DSP
    sos_cmsis = np.zeros((sos.shape[0], 5), dtype=np.float32)
    for i_section, section in enumerate(sos):
        b0, b1, b2, a0, a1, a2 = section
        sos_cmsis[i_section][0] = b0 / a0
        sos_cmsis[i_section][1] = b1 / a0
        sos_cmsis[i_section][2] = b2 / a0
        sos_cmsis[i_section][3] = - a1 / a0
        sos_cmsis[i_section][4] = - a2 / a0
    return sos_cmsis

def coeffs_to_str_c(coeffs):
    def section_to_str(section):
        section_str = ', '.join([str(coeff) + 'f' for coeff in section])
        return section_str
    return '{\n\t' + ',\n\t'.join([section_to_str(section) for section in coeffs]) + '\n}'

if __name__ == "__main__":
    fs = 50
    sos = sig.butter(1, [0.5, 3], btype='bandpass', fs=fs, output='sos')
    sos_cmsis = cmsis_dsp_coeffs(sos)
    print("SciPy")
    print(sos)
    print("CMSIS-DSP")
    print(sos_cmsis)
    print("CMSIS-DSP coeffs C-style arr")
    print(coeffs_to_str_c(sos_cmsis))