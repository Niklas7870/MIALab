import os
import SimpleITK as sitk
import shutil

def main():
    # copy paste complete test folder and add noise

    script_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(script_dir, '../data/test')

    test_loop_parameter = ["", "_gaussian_300","_gaussian_1000","_gaussian_2000","_gaussian_5000",
                           "_salt_pepper_001","_salt_pepper_002","_salt_pepper_005"]

    for test_str in test_loop_parameter:
        save_dir = test_dir + test_str

        if not os.path.exists(save_dir):
            shutil.copytree(test_dir, save_dir)

            parameter = float(test_str.rsplit("_")[-1])

            for subdir, dirs, files in os.walk(save_dir):
                for dir in dirs:
                    imageT1_path = os.path.join(save_dir, dir, 'T1native.nii.gz')
                    imageT2_path = os.path.join(save_dir, dir, 'T2native.nii.gz')

                    imageT1 = sitk.ReadImage(imageT1_path)
                    imageT2 = sitk.ReadImage(imageT2_path)

                    if "gaussian" in save_dir:
                        imageT1_noise = sitk.AdditiveGaussianNoise(imageT1, standardDeviation=parameter, mean=0.0)
                        imageT2_noise = sitk.AdditiveGaussianNoise(imageT2, standardDeviation=parameter, mean=0.0)
                    elif "salt_pepper" in save_dir:
                        if parameter/100 == 0.01:
                            imageT1_noise = sitk.SaltAndPepperNoise(imageT1, probability=0.01, seed=42)
                            imageT2_noise = sitk.SaltAndPepperNoise(imageT2, probability=0.01, seed=42)
                        elif parameter/100 == 0.02:
                            imageT1_noise = sitk.SaltAndPepperNoise(imageT1, probability=0.02, seed=42)
                            imageT2_noise = sitk.SaltAndPepperNoise(imageT2, probability=0.02, seed=42)
                        elif parameter/100 == 0.05:
                            imageT1_noise = sitk.SaltAndPepperNoise(imageT1, probability=0.05, seed=42)
                            imageT2_noise = sitk.SaltAndPepperNoise(imageT2, probability=0.05, seed=42)

                    sitk.WriteImage(imageT1_noise, os.path.join(save_dir, dir, 'T1native.nii.gz'), False)
                    sitk.WriteImage(imageT2_noise, os.path.join(save_dir, dir, 'T2native.nii.gz'), False)

                break  # only go through first level of os.walk

if __name__ == '__main__':
    main()