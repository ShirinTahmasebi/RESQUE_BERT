import os
from sql_metadata import Parser
from utils.utils import *
from imports import *
import sys
sys.path.append('../')


def remove_from_list(input_list, item_to_remove):
    if item_to_remove in input_list:
        input_list.remove(item_to_remove)
    return input_list


def insert_cls_tokens(statement, tables, attributes):
    words = statement.split()

    tables_indices_list = []
    attributes_indices_list = []

    output = []

    for table in tables:
        tables_indices_list = tables_indices_list + find_indices(words, table)

    for attribute in attributes:
        attributes_indices_list = attributes_indices_list + \
            find_indices(words, attribute)

    for index in range(len(words)):
        if index in tables_indices_list:
            output.append('[TBL_CLS]')
        elif index in attributes_indices_list:
            output.append('[ATTR_CLS]')
        output.append(words[index])

    # using the following two assertions, some queries will also be skipped:
    # 1- The ones which can be parsed but they have no from clause.
    # 2- The ones which can be parsed but they have no select clause.
    # However, it may also delete some valid queries in which the select clause uses '*' (rather than attribute names).
    assert '[TBL_CLS]' in output, 'Table name does not exist in the query.'

    # For SQLShare, it is better to comment out this line. The reason is that in SQLShare, many queries are in the form of "SELECT * FROM ..."; so, in case of applying this rule, most of the data will be deleted.
    assert '[ATTR_CLS]' in output, 'There is no attribute in the select clause.'

    return ' '.join(output)


def create_df_with_cls_tokens(raw_data_files_list):
    sessions_with_error = []
    final_df = pd.DataFrame()

    df = pd.read_csv(raw_data_files_list)
    unique_session_id_list = df['session_id'].unique()

    for session_id in unique_session_id_list:

        try:
            single_session_df = df[df['session_id'] == session_id].copy()
            single_session_df['cleaned_statement'] = single_session_df['statement'].apply(
                lambda query: query.replace("$", ""))

            # I decided to comment out this line, because it can cause some problems when the attribute names or the table names include numbers!
            # statements = statements.apply(lambda query: Parser(query).generalize)

            single_session_df['tables'] = single_session_df['cleaned_statement'].apply(
                lambda query: Parser(query).tables
            )

            single_session_df['attributes'] = single_session_df['cleaned_statement'].apply(
                lambda query: remove_from_list(Parser(query).columns, 'NUM')
            )

            single_session_df['statement_with_cls'] = single_session_df[['cleaned_statement', 'tables', 'attributes']]\
                .apply(
                lambda x: insert_cls_tokens(
                    x['cleaned_statement'], x['tables'], x['attributes']
                ),
                axis=1
            )

            final_df = pd.concat(
                [final_df, single_session_df],
                ignore_index=True
            )

        except Exception as ex:
            print(
                f'Error in parsing queries of session: {session_id} - {str(ex)}')
            sessions_with_error = sessions_with_error + [session_id]

    return final_df, sessions_with_error


def execute(raw_path, with_cls_path, name=''):
    with_cls_data_dir_path = get_absolute_path(with_cls_path)
    raw_data_dir_path = get_absolute_path(raw_path)

    final_df, sessions_with_error = create_df_with_cls_tokens(
        raw_data_dir_path)
    final_df.to_csv(with_cls_data_dir_path)

    print("{name} Data with CLS Tokens - Done!")
    print("Files with errors:")
    print(sessions_with_error)


# Preprocess the SDSS dataset
execute(CONSTANTS.DATA_DIR_TRAIN_RAW, CONSTANTS.DATA_DIR_TRAIN_WITH_CLS, 'Train')
execute(CONSTANTS.DATA_DIR_TEST_SDSS_RAW, CONSTANTS.DATA_DIR_TEST_SDSS_WITH_CLS, 'Test')
execute(CONSTANTS.DATA_DIR_VAL_RAW, CONSTANTS.DATA_DIR_VAL_WITH_CLS, 'Validation')

# Preprocess the SQLShare dataset
execute(CONSTANTS.DATA_DIR_TRAIN_SQLSHARE_RAW, CONSTANTS.DATA_DIR_TRAIN_SQLSHARE_WITH_CLS, 'Train')
execute(CONSTANTS.DATA_DIR_TEST_SQLSHARE_RAW, CONSTANTS.DATA_DIR_TEST_SQLSHARE_WITH_CLS, 'Test')
execute(CONSTANTS.DATA_DIR_VAL_SQLSHARE_RAW, CONSTANTS.DATA_DIR_VAL_SQLSHARE_WITH_CLS, 'Validation')

# SDSS:
# Train - Files with error:
# [43538, 60356, 60373, 60388, 122479, 233297, 266006, 293409, 293411, 314938, 414021, 462863, 503574, 503575, 550752, 552242, 676741, 687068, 710453, 744744, 748046, 797233, 860235, 921453, 938883, 940444, 1001715, 1060229, 1060232, 1063581, 1207256, 1207280, 1207290, 1236909, 1237239, 1317802, 1381169, 1391679, 1422728, 1422730, 1422734, 1431672, 1442094, 1458150, 1463927, 1481275, 1546098, 1660795, 1677705, 1693126, 1704781, 1704893, 1713323, 1713324, 1716811, 1716817, 1788556, 1788559, 1793414, 1803731, 2056539, 2087052, 2087074, 2087079, 2279562, 2362357, 2383512, 2394452, 2394504, 2399655, 2399658, 2434699, 2516878, 2646907, 2767909, 2785837, 2785842, 2827790, 2964044, 3030876, 3030985, 3030999, 3049888, 3081825, 3413729, 3413772, 3413773, 3419248, 3453895, 3454343, 3461792, 3528057, 3542504, 3609044, 3714997, 3769612, 3804631, 3846317, 3859177, 3859190, 3930915, 4118031, 4170563, 4170566, 4170574, 4285374, 4401050, 4402282, 4473976, 4507020, 4544312, 4597647, 4598476, 4687311, 4691384, 4703681, 4728843, 4782660, 4862737, 4938115, 5007033, 5015678, 5157827, 5158052, 5245286, 5351274, 5379884, 5397030, 5420265, 5420267, 5433527, 5511441, 5515150, 5552350, 5552540, 5705779, 5768247, 5784695, 5821584, 5821595, 5821599, 5821600, 5847959, 5970842, 5970852, 5970853, 5970855, 5984814, 6288419, 6390678, 6426429, 6521868, 6552287, 6755188, 6763013, 6856320, 7001701, 7016961, 7017603, 7032798, 7032826, 7177525, 7260967, 7284168, 7322071, 7531715, 7553939, 7575255, 7662856, 7662857, 7719456, 7746938, 7771390, 7813078, 7861715, 7922322, 7933263, 8027972, 8186463, 8191909, 8248049, 8310636, 8310775, 8334813, 8356153, 8380654, 8386053, 8404677, 8404681, 8455895, 8520425, 8593799, 8619749, 8619751, 8677259, 8677301, 8677337, 8777032, 8794872, 8794874, 8794885, 8825400, 8965028, 9008353, 9110637, 9110640, 9139629, 9289496, 9295330, 9333249, 9380460, 9419700, 9422983, 9422984, 9440138, 9472939, 9485900, 9727996, 9796250, 9801125, 9870170, 10124158, 10124159, 10124162, 10134313, 10336083, 10378990, 10378998, 10393670, 10417791, 10567485, 10716053, 10785176, 10785177, 10785272, 10785273, 10797228, 10894231, 10937785, 10972517, 10986518, 11035291, 11035307, 11036288, 11137708, 11245329, 11401290, 11401291, 11437229, 11437247, 11509959, 11509961, 11525058, 11601721, 11661662, 11700558, 11712879, 11717303, 11735486, 11857624, 11857625, 11919765, 11932990, 12089985, 12089987, 12158809, 12170498, 12170653, 12170654, 12170660, 12170683, 12170712, 12170812, 12170814, 12170816, 12170991, 12170992, 12170997, 12171098, 12171109, 12171110, 12171114, 12171159, 12171234, 12171659, 12191211, 12191255, 12191286, 12256156, 12576860, 12607664, 12681307, 12681311, 12681318, 12820307, 12820356, 12820429, 12853738, 12853740, 13059711, 13060228, 13092984, 13097793, 13104476, 13226029, 13237297, 13237302, 13275886, 13280989, 13325042, 13383433, 13383437, 13504965, 13523879, 13549122, 13616810, 13647682, 13740929, 13740959, 13781364, 13781554, 13781593, 13792630, 13808918, 13814580, 13827786, 13865248, 13888041, 14010543, 14086263, 14172659, 14247094, 14263440, 14313521, 14490844, 14630580, 14652952, 14665508, 14665509, 14665511, 14665512, 14665519, 14665521, 14665522, 14665523, 14665532, 14665582, 14665583, 14665586, 14665754, 14718755, 14785806, 14827260, 14969621, 14969631, 15029763, 15069497, 15069504, 15282887, 15333354, 15333357, 15356622, 15387338, 15400741, 15400747, 15435932, 15464225, 15478332, 15523629, 15603775, 15617040, 15796817, 15838660, 15848771, 15879774, 15898023, 15928226, 15928227, 15940340, 15966080, 15966093, 16051106, 16059305, 16059306, 16240896, 16265376, 16323262, 16329273, 16341722, 16360846, 16443293, 16445159, 16498582, 16504263, 16517365, 16530247, 16530250, 16580943, 16663959]

# Test - Files with errors:
# [964315, 1063579, 1432992, 1820920, 2087075, 2399660, 2784806, 2827788, 3030859, 3413771, 3551323, 3586082, 3644204, 4768997, 5158059, 6105512, 6390358, 6390676, 7177521, 7531691, 7638020, 7662579, 7797202, 8233190, 8593794, 8619748, 8677331, 8677369, 9247894, 9380458, 10124161, 10336054, 10716047, 10824725, 11137715, 11437227, 11759628, 11885114, 11923043, 12170993, 12170994, 12170999, 12576854, 13523860, 13534608, 13549103, 13549146, 13740952, 13748030, 13882175, 13886295, 14712820, 14938883, 15168033, 15400743, 15400746, 15928231]

# Validation - Files with errors:
# [2894865, 3451498, 4170570, 4198921, 4539691, 5000566, 6121021, 7522807, 7922346, 8404675, 8661686, 9380455, 9420268, 9634052, 11035302, 11245343, 11549931, 11626461, 12516139, 12763347, 13456987, 13781428, 13781739, 13897651, 14628276, 14665516, 14665755, 15562604, 15966085, 16080684]


# SQLShare:
# Train - Files with error:
# (1) [1, 3, 11, 14, 32, 35, 60, 72, 96, 121, 123, 129, 131, 141, 142, 143, 150, 164, 167, 169, 172, 190, 195, 210, 227, 228, 233, 234, 237, 241, 247, 252, 259, 262, 263, 270, 280, 282, 299, 300, 304, 305, 307, 318, 324, 326, 331, 337, 345, 346, 349, 351, 357, 360, 368, 371, 378, 384, 392, 393, 394, 397, 470, 487, 499, 500, 502, 503, 512, 519, 520, 521, 537, 554, 556, 558, 560, 561, 562, 563, 580, 585, 618, 624, 627, 631, 633, 639, 644, 645, 646, 649, 658, 662, 666, 673, 695, 698, 700, 706, 713, 714, 719, 720, 788, 801, 872, 887, 888, 904, 936, 954, 956, 964, 969, 979, 983, 984, 1018, 1020, 1025, 1029, 1030, 1034, 1042, 1043, 1058, 1067, 1087, 1092, 1098, 1103, 1104, 1117, 1119, 1133, 1135, 1136, 1139, 1141, 1142, 1144, 1150, 1167, 1172, 1203, 1208, 1209, 1212, 1213, 1222, 1233, 1235, 1285, 1310, 1320, 1321, 1326, 1329, 1347, 1359, 1360, 1361, 1364, 1396, 1415, 1416, 1447, 1452, 1463, 1535, 1540, 1589, 1590, 1634, 1643, 1646, 1648, 1649, 1650, 1654, 1657, 1661, 1677, 1683, 1684, 1686, 1732, 1736, 1751, 1760, 1767, 1776, 1779, 1789, 1819, 1823, 1835, 1836, 1838, 1865, 1879, 1915, 1920, 1925, 1931, 1943, 1944, 1951, 1956, 2014, 2035, 2036, 2051, 2052, 2085, 2086, 2098, 2114, 2124, 2127, 2163, 2174, 2195, 2226, 2231, 2235, 2264, 2268, 2271, 2273, 2274, 2279, 2295, 2325, 2337, 2339, 2357, 2379, 2382, 2389, 2411, 2415, 2423, 2443, 2457, 2461, 2472, 2495, 2497, 2515, 2520, 2523, 2526, 2546, 2548, 2549, 2555, 2563, 2566, 2569, 2579, 2589, 2602, 2621, 2634, 2639, 2642, 2643, 2648, 2649, 2660, 2671, 2695]
# (2) [3, 60, 96, 123, 129, 131, 141, 142, 143, 169, 210, 228, 237, 252, 259, 262, 263, 270, 299, 300, 305, 307, 318, 326, 331, 346, 351, 357, 360, 368, 378, 470, 487, 499, 500, 502, 503, 512, 519, 520, 521, 585, 624, 631, 633, 644, 645, 649, 658, 662, 666, 673, 695, 713, 714, 719, 788, 872, 887, 888, 954, 956, 964, 979, 983, 1018, 1030, 1058, 1098, 1103, 1104, 1117, 1133, 1135, 1139, 1142, 1150, 1167, 1172, 1203, 1208, 1285, 1320, 1326, 1329, 1347, 1359, 1360, 1361, 1364, 1396, 1415, 1416, 1535, 1643, 1677, 1686, 1732, 1736, 1760, 1767, 1776, 1779, 1865, 1879, 1925, 1931, 1944, 1956, 2085, 2086, 2098, 2195, 2264, 2268, 2337, 2339, 2357, 2379, 2382, 2389, 2411, 2415, 2423, 2443, 2461, 2472, 2520, 2526, 2548, 2549, 2563, 2642, 2648, 2649]


# Test - Files with errors:
# (1) [33, 58, 71, 159, 219, 223, 253, 269, 293, 338, 341, 343, 347, 372, 486, 496, 622, 958, 975, 981, 1041, 1124, 1138, 1146, 1216, 1220, 1223, 1466, 1678, 1787, 2164, 2167, 2521, 2558]
# (2) [219, 223, 253, 269, 293, 343, 347, 486, 496, 958, 981, 1124, 1138, 1146, 1678, 2521]

# Validation - Files with errors:
# (1) [88, 93, 215, 292, 310, 471, 484, 581, 582, 866, 957, 1046, 1059, 1121, 1210, 1225, 1350, 1363, 1647, 1706, 1869, 1884, 1892, 1930, 2037, 2292, 2323, 2338, 2416, 2456, 2529, 2564]

# (2) [88, 215, 310, 471, 484, 957, 1059, 1121, 1225, 1350, 1647, 1706, 1869, 1884, 1930, 2323, 2338, 2416, 2564]