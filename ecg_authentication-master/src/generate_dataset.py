from ecg_authentication import ecg_authentication
from ecg_framework import ecg_framework


def main():
    # ecg_authenticate = ecg_authentication(template_method='template_5_set')
    ecg_authenticate = ecg_authentication(
        template_method='template_avg',
        discrete_wavelet_type='sym4',
        continuous_wavelet='gaus1',
        continuous_wt_scale=50)

    ecg_framework_object = ecg_framework()

    users = 2

    ##########################################################################
    #
    #  ENROLL USER FROM FRAMEWORK
    #  ecg_authenticate = ecg_authenticator object
    #  which_db = "ecg_id" or "ptb" or "qt_db"
    #  qrs_method = "wfdb" or "pantompkins" (recommended wfdb)
    #  no_of_users = number of user you want to enroll
    #  which_filer = "bandpass" or "filfit" or "FIR" or "kalman"
    #
    ##########################################################################

    each_user_template = ecg_framework_object.enroll_user_framework(
        ecg_authenticate=ecg_authenticate,
        which_db="ecg_id",
        qrs_method='wfdb',
        no_of_users=users,
        which_filter="bandpass")


if __name__ == "__main__":
    main()
