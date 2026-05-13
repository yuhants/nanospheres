import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import welch
import scipy.io as sio

def get_area_psd(phi, prefix, prefix2, nfile, channel='C', passband=(60000, 80000)):
    """Calculate PSD then integrate over passband"""

    areas = np.empty(nfile, dtype=np.float32)
    
    for i in range(nfile):
        # fname = f"{prefix}{prefix2}{phi}{prefix2}{phi}_{i+1:02d}.mat"
        # fname = f"{prefix}{prefix2}{phi}{prefix2}{phi}_{i+1:01d}.mat"
        fname = f"{prefix}{prefix2}{phi}.mat"
        data = sio.loadmat(fname)
        
        fs = int(np.floor(1 / data['Tinterval'][0, 0]))
        nperseg = fs / 100
        ff, pp = welch(data[channel][:,0], fs=fs, nperseg=nperseg)

        # print(phi)
        # plt.plot(ff, pp)
        # plt.yscale('log')
        # plt.show()

        idx = np.logical_and(ff > passband[0], ff < passband[1])

        areas[i] = np.trapezoid(pp[idx], ff[idx]) / (2 * np.pi)
        # print(areas)

    area, std_area = np.mean(areas), np.std(areas)
    return area, std_area

def main():

    prefix = r"D:\cooling\20251209_pll_tuning_3e-2mbar"
    phiphi = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]

    prefix2_x = r"\x_"
    channel_x = 'A'
    passband_x = (179000, 220000)

    # phiphi = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    prefix2_y = r"\y_"
    channel_y = 'B'
    passband_y = (160000, 210000)

    # phiphi = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    prefix2_z = r"\z_"
    channel_z = 'C'
    passband_z = (40000, 60000)

    nfile = 1

    area_phi_x = np.empty(len(phiphi), dtype=np.float32)
    area_phi_y = np.empty(len(phiphi), dtype=np.float32)
    area_phi_z = np.empty(len(phiphi), dtype=np.float32)

    # std_area_phi = np.empty(len(phiphi), dtype=np.float32)
    for i, phi in enumerate(phiphi):
        area_x, std_area = get_area_psd(phi, prefix, prefix2_x, nfile, channel_x, passband_x)
        area_y, std_area = get_area_psd(phi, prefix, prefix2_y, nfile, channel_y, passband_y)
        area_z, std_area = get_area_psd(phi, prefix, prefix2_z, nfile, channel_z, passband_z)

        area_phi_x[i] = area_x
        area_phi_y[i] = area_y
        area_phi_z[i] = area_z

        # std_area_phi[i] = std_area

    plt.plot(phiphi, area_phi_x/np.max(area_phi_x), 'b.', markersize=12, label='x')
    plt.plot(phiphi, area_phi_y/np.max(area_phi_y), 'r.', markersize=12, label='y')
    plt.plot(phiphi, area_phi_z/np.max(area_phi_z), 'g.', markersize=12, label='z')

    plt.ylabel('Normalized area under peak (a.u.)')
    plt.xlabel('Phase (deg)')
    plt.legend(frameon=True)

if __name__ == '__main__':
    main()
    plt.show()