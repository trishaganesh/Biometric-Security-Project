from ecg_authentication import ecg_authentication
import glob
import numpy as np


class ecg_framework():
    def __get_ecg_id_file_list(self, file_name, no_of_users):
        file_name_list = []
        for i in range(1, no_of_users + 1):
            folder_name = file_name.format(i)
            file_name_list.append(glob.glob(folder_name)[0])

        for it in range(file_name_list.__len__()):
            file_name_list[it] = file_name_list[it].replace('.dat', '')
        return file_name_list

    def __get_ecg_ptb_file_list(self, file_name, no_of_users):
        file_name_list = []
        for i in range(1, no_of_users + 1):
            if i < 10:
                temp_file_name = file_name + 'patient00{0}/*.dat'
            elif i < 100:
                temp_file_name = file_name + 'patient0{0}/*.dat'
            else:
                temp_file_name = file_name + 'patient{0}/*.dat'
            folder_name = temp_file_name.format(i)
            try:
                file_name_list.append(glob.glob(folder_name)[0])
            except BaseException:
                print(folder_name)
                pass

        for it in range(file_name_list.__len__()):
            file_name_list[it] = file_name_list[it].replace('.dat', '')
        return file_name_list

    def __get_ecg_qt_db_list(self, file_name):
        file_name_list = glob.glob(file_name)
        for it in range(file_name_list.__len__()):
            file_name_list[it] = file_name_list[it].replace('.dat', '')
        return file_name_list

    def __get_fantasia_file_list(self, file_name):
        file_name_list = glob.glob(file_name)
        for it in range(file_name_list.__len__()):
            file_name_list[it] = file_name_list[it].replace('.dat', '')
        del file_name_list[5]
        return file_name_list

    def enroll_user_framework(
            self,
            ecg_authenticate,
            which_db,
            qrs_method,
            no_of_users,
            which_filter):
        #######################################################################
        '''
        #
        #  ENROLL USER FROM FRAMEWORK
        #  ecg_authenticate = ecg_authenticator object
        #  which_db = "ecg_id" or "ecg_id_csv" or "qt_db"
        #  qrs_method = "wfdb" or "pantompkins"
        #  no_of_users = number of user you want to enroll
        #  which_filer = "bandpass" or "filfit" or "FIR" or "kalman"
        #
        '''
        #######################################################################
        #######################################################################
        #
        #                             ECG ID DATABASE FROM PHYSIONET in CSV
        #
        #######################################################################
        each_user_template = []
        if "ecg_id_csv" in which_db:
            file_name = 'ecgid/4_ecg_filtered/person_0{0}_rec_1_ecg.csv'
            each_user_template = ecg_authenticate.enroll_users(
                file_name,
                no_of_users,
                is_wfdb=False,
                qrs_method=qrs_method,
                which_filter=which_filter)

        #######################################################################
        #
        #                             QT DATABASE FROM PHYSIONET
        #
        #######################################################################
        elif "qt_db" in which_db:
            file_name = 'qt-database-1.0.0/*.dat'
            file_name_list = self.__get_ecg_qt_db_list(file_name)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                qrs_method=qrs_method,
                which_filter=which_filter)

        #######################################################################
        #
        #                             ECG ID DATABASE FROM PHYSIONET
        #
        #######################################################################
        elif "ecg_id" in which_db:
            file_name = 'src/data/ecg-id-database-1.0.0/Person_{0}/rec_1.dat'
            file_name_list = self.__get_ecg_id_file_list(
                file_name, no_of_users)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                is_ecg_id=1,
                qrs_method=qrs_method,
                which_filter=which_filter)

        #######################################################################
        #
        #                             ECG PTB DATABASE FROM PHYSIONET
        #                               (Need to rename files in DB)
        #                                       DON'T USE
        #
        #######################################################################
        elif "ecg_ptb" in which_db:
            file_name = 'src/data/physionet.org/files/ptbdb/1.0.0/'
            file_name_list = self.__get_ecg_ptb_file_list(
                file_name, no_of_users)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                qrs_method=qrs_method,
                which_filter=which_filter)

        elif "fantasia" in which_db:
            file_name = 'src/data/physionet.org/files/fantasia/1.0.0/*.dat'
            file_name_list = self.__get_fantasia_file_list(file_name)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                qrs_method=qrs_method,
                which_filter=which_filter)
        elif "mitdb" in which_db:
            file_name = 'src/data/physionet.org/files/mitdb/1.0.0/*.dat'
            file_name_list = self.__get_fantasia_file_list(file_name)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                qrs_method=qrs_method,
                which_filter=which_filter)
        elif "cebsdb" in which_db:
            file_name = 'src/data/physionet.org/files/cebsdb/1.0.0/*.dat'
            file_name_list = self.__get_fantasia_file_list(file_name)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                qrs_method=qrs_method,
                which_filter=which_filter)
        elif "CYBHi" in which_db or 'ecg_bg' in which_db:
            file_name = 'src/data/converted/{}/data/*.dat'.format(which_db)
            file_name_list = self.__get_fantasia_file_list(file_name)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                qrs_method=qrs_method,
                which_filter=which_filter)

        return each_user_template

    def learn_threshold(
            self,
            ecg_authenticate,
            each_user_template,
            qrs_method,
            which_filter,
            which_db,
            all_user):
        threshold_list = []
        file_name = ''
        wfdb = True
        if which_db == 'ecg_id':
            file_name = 'src/data/ecg-id-database-1.0.0/Person_{0}/rec_1'
            wfdb = True
            for i in range(1, all_user + 1):
                file_name_list = []
                temp_name = file_name.format(i)
                file_name_list.append(temp_name)
                result = ecg_authenticate.generate_threshold(
                    file_name_list,
                    each_user_template,
                    i - 1,
                    qrs_method=qrs_method,
                    is_wfdb=wfdb,
                    plot=False,
                    which_filter=which_filter)
                # result = ecg_authenticate.authenticate_user(file_name_list,each_user_template,3-1, threshold=2.48, qrs_method=qrs_method, is_wfdb=True, plot=False, which_filter=which_filter)
                threshold_list.append(result)
        elif which_db == 'qt_db':
            file_name = 'src/data/qt-database-1.0.0/*.dat'
            wfdb = True
        elif (which_db == 'fantasia') or (which_db == 'mitdb') or (which_db == 'cebsdb') or (which_db == 'CYBHi') or (which_db == 'ecg_bg'):
            file_name = ''
            if which_db == 'CYBHi' or which_db == 'ecg_bg':
                file_name = 'src/data/converted/{}/data/*.dat'.format(which_db)
            else:
                file_name = 'src/data/physionet.org/files/{0}/1.0.0/*.dat'.format(
                    which_db)
            wfdb = True
            i = 0
            file_name_list_all = self.__get_fantasia_file_list(file_name)
            for each_file in file_name_list_all:
                if i >= all_user:
                    break
                i += 1
                file_name_list = []
                file_name_list.append(each_file)
                result = ecg_authenticate.generate_threshold(
                    file_name_list,
                    each_user_template,
                    i - 1,
                    qrs_method=qrs_method,
                    is_wfdb=wfdb,
                    plot=False,
                    which_filter=which_filter,
                    is_auth=False)
                # result = ecg_authenticate.authenticate_user(file_name_list,each_user_template,3-1, threshold=2.48, qrs_method=qrs_method, is_wfdb=True, plot=False, which_filter=which_filter)
                threshold_list.append(result)
        elif "ecg_ptb" in which_db:
            file_name = 'src/data/physionet.org/files/ptbdb/1.0.0/'
            wfdb = True
            i = 0
            file_name_list_all = self.__get_ecg_ptb_file_list(
                file_name, all_user)
            all_user = file_name_list_all.__len__()
            for each_file in file_name_list_all:
                if i >= all_user:
                    break
                i += 1
                file_name_list = []
                file_name_list.append(each_file)
                result = ecg_authenticate.generate_threshold(
                    file_name_list,
                    each_user_template,
                    i - 1,
                    qrs_method=qrs_method,
                    is_wfdb=wfdb,
                    plot=False,
                    which_filter=which_filter,
                    is_auth=False)
                # result = ecg_authenticate.authenticate_user(file_name_list,each_user_template,3-1, threshold=2.48, qrs_method=qrs_method, is_wfdb=True, plot=False, which_filter=which_filter)
                threshold_list.append(result)

        return threshold_list

    def authenticate_user_framework(
            self,
            ecg_authenticate,
            each_user_template,
            which_db,
            qrs_method,
            which_filter,
            threshold_template,
            all_user=1,
            which_user=1):
        #######################################################################
        '''
        #
        #  Authenticate User
        #  ecg_authenticate = ecg_authenticator object
        #  each_user_template = template generated from enroll user
        #  which_db = "ecg_id" or "ecg_id_csv" or "qt_db"
        #  qrs_method = "wfdb" or "pantompkins"
        #  which_filer = "bandpass" or "filfit" or "FIR" or "kalman"
        #  all_user = 1 -> authenticate with one user
        #           > 1 -> authenticate with all users -> user numbers
        #
        '''
        #######################################################################
        #######################################################################
        #
        #                             ECG ID DATABASE FROM PHYSIONET CSV
        #
        #######################################################################
        if all_user == 1:
            if "ecg_id_csv" in which_db:
                file_name = 'ecgid/4_ecg_filtered/person_03_rec_3_ecg.csv'
                decomposed_data_for_each_segment = ecg_authenticate.authenticate_user(
                    file_name,
                    each_user_template,
                    3 - 1,
                    qrs_method=qrs_method,
                    is_wfdb=False,
                    plot=False,
                    threshold=threshold_template[
                        3 - 1],
                    which_filter=which_filter)
                for each_segment in decomposed_data_for_each_segment[0]:
                    temp_result = ecg_authenticate.step6_matching(
                        each_user_template[3 - 1], each_segment[0], threshold=(threshold_template[user] * threshold_offset), plot=False)
                    if temp_result:
                        result += 1
                pass_val = result / \
                    decomposed_data_for_each_segment[0].__len__()
                if pass_val > 0.5:
                    return True
                else:
                    return False

        #######################################################################
        #
        #                             ECG ID DATABASE FROM PHYSIONET
        #
        #######################################################################
            elif "ecg_id" in which_db:
                file_name_list = []
                result = 0
                test_file = 'src/data/ecg-id-database-1.0.0/Person_{}/rec_1'.format(
                    which_user)
                file_name_list.append(test_file)
                # result = ecg_authenticate.authenticate_user(file_name_list, each_user_template, 3-1, threshold=2.48, qrs_method=qrs_method, is_wfdb=True, plot=False, which_filter=which_filter)
                decomposed_data_for_each_segment = ecg_authenticate.authenticate_user(
                    file_name_list,
                    each_user_template,
                    3 - 1,
                    qrs_method=qrs_method,
                    is_wfdb=True,
                    plot=False,
                    threshold=threshold_template[
                        3 - 1],
                    which_filter=which_filter)
                for each_segment in decomposed_data_for_each_segment[0]:
                    user = which_user - 1
                    threshold_offset = 1
                    temp_result = ecg_authenticate.step6_matching(
                        each_user_template[user], each_segment[0], threshold=(
                            threshold_template[user] * threshold_offset), plot=False)
                    if temp_result:
                        result += 1

                pass_val = result / \
                    decomposed_data_for_each_segment[0].__len__()
                if pass_val > 0.5:
                    return True
                else:
                    return False
            elif "mitdb" in which_db:
                file_name_list = []
                result = 0
                test_file = 'src/data/ecg-id-database-1.0.0/Person_{}/rec_2'.format(
                    which_user)
                file_name_list.append(test_file)
                # result = ecg_authenticate.authenticate_user(file_name_list, each_user_template, 3-1, threshold=2.48, qrs_method=qrs_method, is_wfdb=True, plot=False, which_filter=which_filter)
                decomposed_data_for_each_segment = ecg_authenticate.authenticate_user(
                    file_name_list,
                    each_user_template,
                    3 - 1,
                    qrs_method=qrs_method,
                    is_wfdb=True,
                    plot=False,
                    threshold=threshold_template[
                        3 - 1],
                    which_filter=which_filter,
                    is_auth=False)
                for each_segment in decomposed_data_for_each_segment[0]:
                    user = which_user - 1
                    threshold_offset = 1
                    temp_result = ecg_authenticate.step6_matching(
                        each_user_template[user], each_segment[0], threshold=(
                            threshold_template[user] * threshold_offset), plot=False)
                    if temp_result:
                        result += 1

                pass_val = result / \
                    decomposed_data_for_each_segment[0].__len__()
                if pass_val > 0.5:
                    return True
                else:
                    return False

        #######################################################################
        #
        #                             Authenticate all users
        #
        #######################################################################
        else:
            if "ecg_id" in which_db:
                ret = []
                file_name = 'src/data/ecg-id-database-1.0.0/Person_{0}/rec_1'
                threshold_iterate = 1
                step = 0.1
                start_thresh = 1.0
                end_thresh = start_thresh + ((step * threshold_iterate) - step)
                correct_user = [0] * threshold_iterate
                false_user = [0] * threshold_iterate
                ret_result = []
                for j in range(1, all_user + 1):
                    for i in range(1, all_user + 1):
                        file_name_list = []
                        temp_name = file_name.format(i)
                        file_name_list.append(temp_name)
                        user = j - 1
                        # print("---------------------------------------------------------------------------------------------------")
                        # thresh = (threshold_template[user]*1.1)
                        thresh = 10
                        decomposed_data_for_each_segment = ecg_authenticate.authenticate_user(
                            file_name_list,
                            each_user_template,
                            user,
                            qrs_method=qrs_method,
                            is_wfdb=True,
                            plot=False,
                            threshold=thresh,
                            which_filter=which_filter)
                        for threshold_offset in np.arange(1.0, 1.1, 1):
                            if decomposed_data_for_each_segment[0].__len__(
                            ) == 0:
                                continue
                            # for threshold_offset in
                            # np.arange(start_thresh,end_thresh,step):
                            result = 0
                            f_method = 0
                            if ecg_authenticate.feature_method == 'fiducial':
                                f_method = 1
                            for each_segment in decomposed_data_for_each_segment[0]:
                                temp_result = ecg_authenticate.step6_matching(
                                    each_user_template[user], each_segment[f_method], threshold=(
                                        threshold_template[user] * threshold_offset), plot=False)
                                if temp_result:
                                    result += 1

                            # result = ecg_authenticate.authenticate_user(file_name_list,each_user_template,3-1, threshold=2.48, qrs_method=qrs_method, is_wfdb=True, plot=False, which_filter=which_filter)
                            # print("---------------------------------------------------------------------------------------------------")
                            pass_val = result / \
                                decomposed_data_for_each_segment[0].__len__()
                            t_i = 0
                            for pass_threshold in np.arange(0, 1, 1):
                                if pass_val >= pass_threshold:
                                    if(i == j):
                                        # print("User is authenticated correctly = " + str(i))
                                        correct_user[t_i] += 1
                                    else:
                                        # print("User is authenticated incorrectly = " + str(j) + " as = " + str(i))
                                        false_user[t_i] += 1
                                t_i += 1
                            # ret_result.append([pass_threshold,correct_user,false_user)
                    # print(ret_result)

                for res in range(threshold_iterate):
                    accuracy = (correct_user[res] / all_user) * 100
                    false_accept_rate = (
                        false_user[res] / (all_user * all_user)) * 100
                    # print("Threshold = " + str(threshold_offset))
                    # print("Accuracy = " + str(accuracy))
                    # print("False Accept Rate = " + str(false_accept_rate))
                    accuracy_list = [res, accuracy, false_accept_rate]
                    ret.append(accuracy_list)
                return ret

            elif ("fantasia" in which_db) or ("mitdb" in which_db) or (which_db == 'cebsdb') or (which_db == 'ecg_ptb') or (which_db == 'CYBHi') or (which_db == 'ecg_bg'):
                ret = []
                unused_user = 0
                # file_name = 'ecg-id-database-1.0.0/Person_{0}/rec_1'
                threshold_iterate = 1
                step = 0.1
                start_thresh = 1.0
                end_thresh = start_thresh + ((step * threshold_iterate) - step)
                correct_user = [0] * threshold_iterate
                false_user = [0] * threshold_iterate
                ret_result = []
                file_name = ''
                file_name_list_all = []
                if which_db == 'ecg_ptb':
                    file_name = 'src/data/physionet.org/files/ptbdb/1.0.0/'
                    file_name_list_all = self.__get_ecg_ptb_file_list(
                        file_name, all_user)
                    all_user = file_name_list_all.__len__()
                elif which_db == 'CYBHi' or which_db == 'ecg_bg':
                    file_name = 'src/data/converted/{}/data/*.dat'.format(
                        which_db)
                    file_name_list_all = self.__get_fantasia_file_list(
                        file_name)
                else:
                    file_name = 'src/data/physionet.org/files/{0}/1.0.0/*.dat'.format(
                        which_db)
                    file_name_list_all = self.__get_fantasia_file_list(
                        file_name)
                start_t = 1.0
                end_t = 1.09
                # if ("mitdb" in which_db):
                #     start_t = 1.2
                #     end_t = 1.29

                for j in range(1, all_user + 1):
                    i = -1
                    for temp_name in file_name_list_all:
                        if i >= all_user:
                            break
                        i += 1
                        file_name_list = []
                        file_name_list.append(temp_name)
                        user = j - 1
                        # print("---------------------------------------------------------------------------------------------------")
                        # thresh = (threshold_template[user]*1.1)
                        thresh = 10
                        decomposed_data_for_each_segment = ecg_authenticate.authenticate_user(
                            file_name_list,
                            each_user_template,
                            user,
                            qrs_method=qrs_method,
                            is_wfdb=True,
                            plot=False,
                            threshold=thresh,
                            which_filter=which_filter,
                            is_auth=False)

                        for threshold_offset in np.arange(start_t, end_t, 0.1):
                            # for threshold_offset in
                            # np.arange(start_thresh,end_thresh,step):
                            result = 0
                            f_method = 0
                            skipped_segment = 0
                            if ecg_authenticate.feature_method == 'fiducial':
                                f_method = 1
                            if decomposed_data_for_each_segment[0].__len__(
                            ) == 0:
                                continue
                            for each_segment in decomposed_data_for_each_segment[0]:
                                if ecg_authenticate.feature_method == 'fiducial':
                                    if (each_segment.__len__() < 2) or (each_user_template[user].__len__() == 0) or (
                                            threshold_template[user] == np.NaN):
                                        skipped_segment += 1
                                        continue
                                else:
                                    if (each_user_template[user][0].__len__() == 0) or (
                                            threshold_template[user] == np.NaN):
                                        skipped_segment += 1
                                        continue
                                temp_result = ecg_authenticate.step6_matching(
                                    each_user_template[user], each_segment[f_method], threshold=(
                                        threshold_template[user] * threshold_offset), plot=False)
                                if temp_result:
                                    result += 1

                            # result = ecg_authenticate.authenticate_user(file_name_list,each_user_template,3-1, threshold=2.48, qrs_method=qrs_method, is_wfdb=True, plot=False, which_filter=which_filter)
                            # print("---------------------------------------------------------------------------------------------------")
                            denominator_pass = (
                                decomposed_data_for_each_segment[0].__len__() - skipped_segment)
                            if denominator_pass > 0:
                                pass_val = result / denominator_pass
                            else:
                                pass_val = 0
                            t_i = 0
                            for pass_threshold in np.arange(0, 1, 1):
                                if pass_val > pass_threshold:
                                    if(i == user):
                                        # print("User is authenticated correctly = " + str(i))
                                        correct_user[t_i] += 1
                                    else:
                                        # print("User is authenticated incorrectly = " + str(j) + " as = " + str(i))
                                        false_user[t_i] += 1
                                        if t_i > 0:
                                            print("d")
                                t_i += 1
                            # ret_result.append([pass_threshold,correct_user,false_user)
                    # print(ret_result)

                for res in range(threshold_iterate):
                    accuracy = (correct_user[res] / all_user) * 100
                    false_accept_rate = (
                        false_user[res] / ((all_user * all_user))) * 100
                    # print("Threshold = " + str(threshold_offset))
                    # print("Accuracy = " + str(accuracy))
                    # print("False Accept Rate = " + str(false_accept_rate))
                    accuracy_list = [res, accuracy, false_accept_rate]
                    ret.append(accuracy_list)
                return ret
                # pass
