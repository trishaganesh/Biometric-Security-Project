
from ecg_authentication import ecg_authentication
from ecg_framework import ecg_framework


def main():
    # ecg_authenticate = ecg_authentication(template_method='template_5_set')
    discrete_wavelet_type = 'db4'
    ecg_authenticate = ecg_authentication(
        template_method='template_avg',
        discrete_wavelet_type=discrete_wavelet_type,
        continuous_wavelet='gaus1',
        continuous_wt_scale=50,
        feature_method='non-fiducial',
        segmentation_method="window")

    ecg_framework_object = ecg_framework()

    users = 2

    ##########################################################################
    #
    #  ENROLL USER FROM FRAMEWORK
    #  ecg_authenticate = ecg_authenticator object
    #  which_db = "ecg_id" or "ecg_id_csv" or "qt_db" fantasia, cebsdb, ecg_ptb, 'CYBHi', 'ecg_bg
    #  qrs_method = "wfdb" or "pantompkins"
    #  no_of_users = number of user you want to enroll
    #  which_filer = "bandpass" or "filfit" or "FIR" or "kalman"
    #  segmentation_method = "window" or RR
    #
    ##########################################################################
    # print("Enrolling")
    qrs = 'pantompkins'
    filter = 'FIR'
    db = 'ecg_id'

    each_user_template = ecg_framework_object.enroll_user_framework(
        ecg_authenticate=ecg_authenticate,
        which_db=db,
        qrs_method=qrs,
        no_of_users=users,
        which_filter=filter
    )
    ##########################################################################
    #
    #  Update Templates
    #
    ##########################################################################

    # file_name = 'ecgid/4_ecg_filtered/person_03_rec_3_ecg.csv'
    # ecg_authenticate.step7_template_update(each_user_template,file_name,3-1)
    # print("Learn Threshold")
    threshold_list = ecg_framework_object.learn_threshold(
        ecg_authenticate=ecg_authenticate,
        each_user_template=each_user_template,
        qrs_method=qrs,
        which_db=db,
        which_filter=filter,
        all_user=users)

    ##########################################################################
    #
    #  Authenticate User
    #  ecg_authenticate = ecg_authenticator object
    #  each_user_template = template generated from enroll user
    #  which_db = "ecg_id" or "ecg_id_csv" or "qt_db"
    #  qrs_method = "wfdb" or "pantompkins"
    #  which_filer = "bandpass" or "filfit" or "FIR" or "kalman"
    #  all_user = 1 -> authenticate with one user
    #           > 1 = users -> authenticate with all users -> user numbers
    #
    ##########################################################################
    # print("Authenticate")
    write_file_name = 'src/Result/' + db + '_' + str(users) + '_' + qrs + '_' + filter + '_' + discrete_wavelet_type + \
        '_' + ecg_authenticate.feature_method + '_' + ecg_authenticate.segmentation_method + '.csv'
    dump_op = open(write_file_name, "w")

    result = ecg_framework_object.authenticate_user_framework(
        ecg_authenticate=ecg_authenticate,
        each_user_template=each_user_template,
        which_db=db,
        qrs_method=qrs,
        which_filter=filter,
        threshold_template=threshold_list,
        all_user=users)
    for each_result in result:
        print(str(each_result[0]) + "," +
              str(each_result[1]) + "," + str(each_result[2]))
        dump_op.write(str(each_result[0]) +
                      "," +
                      str(each_result[1]) +
                      "," +
                      str(each_result[2]))
        dump_op.write('\n')

    # result = ecg_framework_object.authenticate_user_framework(ecg_authenticate=ecg_authenticate,each_user_template=each_user_template
    #                                         , which_db="ecg_id", qrs_method='wfdb',which_filter="bandpass"
    #                                         , threshold_template=threshold_list, all_user=1, which_user=4)

    # if result:
    #     print("User Authenticated\n")

    # print(result)


if __name__ == "__main__":
    main()
