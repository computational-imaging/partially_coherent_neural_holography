def spectra_measured(source_type='LED', ch=None):
    spectra_filename = f'./spectra/{source_type}.txt'
    spectra = {}
    for line in open(spectra_filename, "r"):
        split_sym = ","
        max_spectrum = 1.

        cols = line.split(split_sym)
        wv = float(cols[0])
        if wv < 500:
            ch = 2
        elif wv < 600:
            ch = 1
        elif wv < 700:
            ch = 0
        spectra[wv] = abs(float(cols[ch+1])) / max_spectrum
    return spectra

def wvl2transmission_measured(wavelength, source_type='LED'):
    wv2trans_measured = spectra_measured(source_type)
    t = list(wv2trans_measured.keys())
    wavelength = wavelength * 1e9 # in nm
    lower = max([i for i in t if i < wavelength])
    upper = min([i for i in t if i > wavelength])

    # interpolate
    if upper == lower:
        transmission = wv2trans_measured[lower]
    else:
        transmission = ((upper - wavelength) * wv2trans_measured[lower] \
                       + (wavelength - lower) * wv2trans_measured[upper]) / (upper - lower)

    return transmission