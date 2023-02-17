def execute(list_of_templatized_dfs):

    print('Statistics of  the Dataset Templates!')

    for i in range(len(list_of_templatized_dfs)):
        for j in range(i+1, len(list_of_templatized_dfs)):
            name_1 = list_of_templatized_dfs[i]['name']
            name_2 = list_of_templatized_dfs[j]['name']

            df_1 = list_of_templatized_dfs[i]['df']
            df_2 = list_of_templatized_dfs[j]['df']

            template_set_df_1 = set(df_1['template_v2'])
            template_set_df_2 = set(df_2['template_v2'])

            difference_1_2 = set(template_set_df_1 - template_set_df_2)
            difference_2_1 = set(template_set_df_2 - template_set_df_1)
            intersection = set(template_set_df_2 - difference_2_1)

            list_of_frequencies_1_2 = []
            for template in difference_1_2:
                list_of_frequencies_1_2.append(
                    len(df_1[df_1['template_v2'] == template])
                )

            list_of_frequencies_2_1 = []
            for template in difference_2_1:
                list_of_frequencies_2_1.append(
                    len(df_2[df_2['template_v2'] == template])
                )

            print("--------------------------------------------------")
            print(f"DF1 = {name_1}, DF2 = {name_2}")
            print("--------------------------------------------------")
            print(f"Unique templates in DF1 = {len(template_set_df_1)}")
            print(f"Unique templates in DF2 = {len(template_set_df_2)}")
            print(f"DF1 - DF2 = {len(difference_1_2)}")
            print(f"DF2 - DF1 = {len(difference_2_1)}")
            print(f"Intersection between DF1 and DF2 = {len(intersection)}")
            print("---> Unique Template in DF1")
            print(f"Total = {sum(list_of_frequencies_1_2)}")
            print(f"Max = {max(list_of_frequencies_1_2)}")
            print(f"Min = {min(list_of_frequencies_1_2)}")
            print(
                f"Mean = {sum(list_of_frequencies_1_2) / len(list_of_frequencies_1_2)}")
            print(
                f"Percentage = {sum(list_of_frequencies_1_2) / len(template_set_df_1)}")
            print("---> Unique Template in DF2")
            print(f"Total = {sum(list_of_frequencies_2_1)}")
            print(f"Max = {max(list_of_frequencies_2_1)}")
            print(f"Min = {min(list_of_frequencies_2_1)}")
            print(
                f"Mean = {sum(list_of_frequencies_2_1) / len(list_of_frequencies_2_1)}")
            print(
                f"Percentage = {sum(list_of_frequencies_2_1) / len(template_set_df_2)}")
