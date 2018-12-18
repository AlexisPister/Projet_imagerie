import nibabel as nib
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


path = "./Images_Concentration/ConcentrationImage_patient_048.nii"

img = nib.load(path)
img.shape
img_data = img.get_fdata()

niiftis = glob.glob("Images_Concentration/*")
masks = glob.glob('Masques/*')
niiftis.sort()
masks.sort()

# Liste contenant tous les pixels des 4 patients
patients_GM = []
patients_WM = []
patients_TUM = []

#%%
for nii, mask in zip(niiftis, masks):
    print(nii, mask)
    img = nib.load(nii)
    img_data = img.get_fdata()
    mask = nib.load(mask)
    mask_data = mask.get_fdata()
    GM_inds = np.argwhere(mask_data == 1)
    WM_inds = np.argwhere(mask_data == 2)
    TUM_inds = np.argwhere(mask_data == 3)
    print(GM_inds)

    # Sauvegarde de l'evolution des intensités des voxels pour les 3 tissus dans des arrays (nb voxels, 60)
    for i,inds in enumerate(WM_inds[:10000]):
        voxels = img_data[tuple(inds)]
        if i == 0:
            all_voxels_WM = voxels
        else:
            all_voxels_WM = np.vstack((all_voxels_WM, voxels))
    for i,inds in enumerate(GM_inds[:10000]):
        voxels = img_data[tuple(inds)]
        if i == 0:
            all_voxels_GM = voxels
        else:
            all_voxels_GM = np.vstack((all_voxels_GM, voxels))
    for i,inds in enumerate(TUM_inds[:10000]):
        voxels = img_data[tuple(inds)]
        if i == 0:
            all_voxels_TUM = voxels
        else:
            all_voxels_TUM = np.vstack((all_voxels_TUM, voxels))

    patients_GM.append(all_voxels_GM)
    patients_WM.append(all_voxels_WM)
    patients_TUM.append(all_voxels_TUM)

    # Distributions moyennes des 3 tissus
    dist_mean_WM = np.apply_along_axis(np.mean, 0, all_voxels_WM)
    dist_mean_GM = np.apply_along_axis(np.mean, 0, all_voxels_GM)
    dist_mean_TUM = np.apply_along_axis(np.mean, 0, all_voxels_TUM)

    # Plot
    plt.plot(dist_mean_GM)
    plt.plot(dist_mean_WM)
    plt.plot(dist_mean_TUM)
    plt.xlabel("t")
    plt.ylabel("Intensité")
    plt.legend(["Matière grise", "Matière blanche", "tumeur"])
    plt.show()

    break

#%% Descente de gradient
def Xp(t, tmax, a, d):
    return np.nan_to_num(np.exp(a * (1 - (t - d)/tmax)))

def T(t, tmax, d):
    return np.nan_to_num((t - d) / tmax)

def f(t, tmax, ymax, a, d):
    return np.nan_to_num(ymax * T(t, d, a)**a * Xp(t, tmax, a, d) * (t > d))

def dfdtmax(t, tmax, ymax, a, d):
    return np.nan_to_num(ymax * a * Xp(t, tmax, a, d) * (t-d)**a / (tmax)**(a+1) * (T(t, tmax, d) - 1) * (t > d))

def dfdymax(t, tmax, ymax, a, d):
    return np.nan_to_num(T(t, tmax, d)**a * Xp(t, tmax, a, d) * (t > d))

def dfdd(t, tmax, ymax, a, d):
    return np.nan_to_num(a * ymax * T(t, tmax, d)**(a-1) / tmax * Xp(t, tmax, a ,d) * (T(t, tmax, d) - 1) * (t > d))

def dfda(t, tmax, ymax, a, d):
    return np.nan_to_num(ymax * T(t, tmax, d) * Xp(t, tmax, a, d) * (np.log(T(t, tmax, d) + 1 - T(t, tmax, d))) * (t > d))

def SCE(Y, t, tmax, ymax, a, d):
    return np.nan_to_num(0.5 * np.sum((Y - f(t, tmax, ymax, a, d))**2))

def dSCEdtmax(Y, t, tmax, ymax, a, d):
    return np.nan_to_num(-np.sum((Y - f(t, tmax, ymax, a, d)) * dfdtmax(t, tmax, ymax, a, d)))

def dSCEdymax(Y, t, tmax, ymax, a, d):
    return np.nan_to_num(-np.sum((Y - f(t, tmax, ymax, a, d)) * dfdymax(t, tmax, ymax, a, d)))

def dSCEdd(Y, t, tmax, ymax, a, d):
    return np.nan_to_num(-np.sum((Y - f(t, tmax, ymax, a, d)) * dfdd(t, tmax, ymax, a, d)))

def dSCEda(Y, t, tmax, ymax, a, d):
    return np.nan_to_num(-np.sum((Y - f(t, tmax, ymax, a, d)) * dfda(t, tmax, ymax, a, d)))

# Condition initiale
tmax = 20
ymax = 6
a = 2
d = 3
params = np.array([tmax, ymax, a, d])
pas = 10**(-13)

Y = all_voxels_WM[10]
times = np.arange(0, 60)

i = 0
while i < 5:
    tmax = params[0]
    ymax = params[1]
    a = params[2]
    d = params[3]

    print(SCE(Y, times, tmax, ymax, a, d))

    # Derive partielles de la SCE
    dtmax = dSCEdtmax(Y, times, tmax, ymax, a, d)
    dymax = dSCEdymax(Y, times, tmax, ymax, a, d)
    da = dSCEda(Y, times, tmax, ymax, a, d)
    dd = dSCEdd(Y, times, tmax, ymax, a, d)

    # Gradient
    grad = np.array([dtmax, dymax, da, dd])

    # Derives partielles de f (loi gamma)
    dftmax = dfdtmax(times, tmax, ymax, a, d)
    dfymax = dfdymax(times, tmax, ymax, a, d)
    dfa = dfda(times, tmax, ymax, a, d)
    dfd = dfdd(times, tmax, ymax, a, d)



# =============================================================================
#     # Hessienne
#     hess = np.array([[np.sum(dftmax * dftmax), np.sum(dftmax * dfymax),np.sum(dftmax * dfa),np.sum(dftmax * dfd)],
#                      [np.sum(dfymax * dftmax), np.sum(dfymax * dfymax),np.sum(dfymax * dfa),np.sum(dfymax * dfd)],
#                      [np.sum(dfa * dftmax), np.sum(dfa * dfymax),np.sum(dfa * dfa),np.sum(dfa * dfd)],
#                      [np.sum(dfd * dftmax), np.sum(dfd * dfymax),np.sum(dfd * dfa),np.sum(dfd * dfd)]])
#     hess_inv = np.linalg.inv(hess)
#     dist = -np.dot(grad, hess_inv)
#     print(dist)
# =============================================================================

    # Descente
    # Ordre 1
    params = params - pas * grad

    # Ordre 2
    #params = params + pas * dist
    print(params)

    i += 1









