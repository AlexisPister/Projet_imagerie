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


