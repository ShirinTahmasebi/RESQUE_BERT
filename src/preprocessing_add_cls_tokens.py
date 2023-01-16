import sys
sys.path.append('../')

from imports import *
from utils.utils import *
from sql_metadata import Parser

import os 

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
        attributes_indices_list = attributes_indices_list + find_indices(words, attribute)

    for index in range(len(words)):
        if index in tables_indices_list:
            output.append('[TBL_CLS]')
        elif index in attributes_indices_list:
            output.append('[ATTR_CLS]')
        output.append(words[index])
    
    return ' '.join(output)


def create_df_with_cls_tokens(raw_data_files_list):
    sessions_with_error = []
    final_df = pd.DataFrame()

    df = pd.read_csv(raw_data_files_list)
    unique_session_id_list = df['session_id'].unique()

    for session_id in unique_session_id_list:
        try:
            single_session_df = df[df['session_id'] == session_id].copy()
            statements = single_session_df['statement'].apply(lambda query: query.replace("$", ""))
            statements = statements.apply(lambda query: Parser(query).generalize)

            single_session_df['tables'] = statements.apply(lambda query: Parser(query).tables)
            single_session_df['attributes'] = statements.apply(lambda query: remove_from_list(Parser(query).columns, 'NUM'))

            single_session_df['statement_with_cls'] = single_session_df[['statement', 'tables', 'attributes']].apply(
                lambda x: insert_cls_tokens(x['statement'], x['tables'], x['attributes']), 
                axis=1
            )
            
            final_df = pd.concat([final_df, single_session_df], ignore_index=True)

        except Exception as ex:
            print(f'Error in parsing queries of session: {session_id} - {str(ex)}')
            sessions_with_error = sessions_with_error + [session_id]
    
    return final_df, sessions_with_error



def execute(raw_path, with_cls_path, name=''):
    with_cls_data_dir_path = get_absolute_path(with_cls_path)
    raw_data_dir_path = get_absolute_path(raw_path)

    final_df, sessions_with_error = create_df_with_cls_tokens(raw_data_dir_path)
    final_df.to_csv(with_cls_data_dir_path)

    print("{name} Data with CLS Tokens - Done!")
    print("Files with errors:")
    print(sessions_with_error)


execute(CONSTANTS.DATA_DIR_TRAIN_RAW, CONSTANTS.DATA_DIR_TRAIN_WITH_CLS, 'Train')
execute(CONSTANTS.DATA_DIR_TEST_RAW, CONSTANTS.DATA_DIR_TEST_WITH_CLS, 'Test')
execute(CONSTANTS.DATA_DIR_VAL_RAW, CONSTANTS.DATA_DIR_VAL_WITH_CLS, 'Validation')

# Train - Files with error:
# [43538, 60356, 60373, 60388, 122479, 233297, 266006, 314938, 414021, 462863, 503574, 503575, 687068, 710453, 940444, 1317802, 1422728, 1422730, 1422734, 1458150, 1463927, 1481275, 1546098, 1677705, 1704781, 1713323, 1716817, 1793414, 1803731, 2056539, 2087052, 2087074, 2087079, 2394504, 2399655, 2399658, 2516878, 2646907, 2767909, 2785837, 2785842, 2827790, 3030985, 3049888, 3081825, 3413772, 3413773, 3454343, 3528057, 3714997, 3846317, 3859177, 3859190, 4118031, 4170574, 4285374, 4401050, 4473976, 4544312, 4597647, 4598476, 4691384, 4728843, 5007033, 5157827, 5245286, 5351274, 5379884, 5397030, 5420265, 5420267, 5511441, 5552350, 5552540, 5784695, 5821584, 5970842, 5970853, 5970855, 6288419, 6390678, 6552287, 6755188, 6763013, 6856320, 7001701, 7284168, 7531715, 7719456, 7746938, 7861715, 7922322, 7933263, 8186463, 8248049, 8334813, 8380654, 8386053, 8404677, 8404681, 8455895, 8520425, 8619749, 8619751, 8677301, 8677337, 8777032, 8794872, 8794874, 8794885, 9008353, 9110637, 9110640, 9139629, 9289496, 9380460, 9419700, 9422983, 9422984, 9440138, 9472939, 9485900, 9727996, 9801125, 9870170, 10124158, 10124159, 10124162, 10134313, 10336083, 10378990, 10393670, 10417791, 10567485, 10785176, 10785177, 10785272, 10785273, 10937785, 10972517, 10986518, 11035291, 11035307, 11036288, 11245329, 11401291, 11509959, 11525058, 11601721, 11700558, 11717303, 11735486, 11932990, 12089985, 12089987, 12158809, 12170712, 12170991, 12170992, 12170997, 12171110, 12820429, 13060228, 13092984, 13104476, 13226029, 13237302, 13523879, 13549122, 13740929, 13740959, 13781554, 13792630, 13808918, 13827786, 14010543, 14086263, 14263440, 14313521, 14490844, 14630580, 14665508, 14665509, 14665511, 14665512, 14665519, 14665521, 14665522, 14665523, 14665754, 14718755, 15029763, 15282887, 15796817, 15879774, 15898023, 15966080, 15966093, 16051106, 16059305, 16059306, 16341722, 16443293, 16445159, 16504263, 16530247, 16530250, 16580943]

# Test - Files with errors:
# [964315, 1820920, 2087075, 2399660, 2784806, 2827788, 3413771, 3586082, 5158059, 6390358, 6390676, 7531691, 7662579, 8619748, 8677331, 9380458, 10336054, 10824725, 11137715, 11759628, 11923043, 12170993, 12170994, 12576854, 13523860, 13740952, 13886295, 14712820, 14938883]

# Validation - Files with errors:
# [2894865, 3451498, 4170570, 4198921, 4539691, 5000566, 6121021, 7522807, 7922346, 8404675, 8661686, 9380455, 9420268, 9634052, 11035302, 11245343, 11549931, 11626461, 12516139, 12763347, 13456987, 13781428, 13781739, 13897651, 14628276, 14665516, 14665755, 15562604, 15966085, 16080684]



