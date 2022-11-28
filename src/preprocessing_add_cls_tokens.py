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


def write_files_with_cls_tokens(raw_data_files_list):
    files_with_error = []
    for data_file in raw_data_files_list:
        try:
            df = pd.read_csv(data_file)
            statements = df['statement'].apply(lambda query: query.replace("$", ""))
            statements = statements.apply(lambda query: Parser(query).generalize)

            df['tables'] = statements.apply(lambda query: Parser(query).tables)
            df['attributes'] = statements.apply(lambda query: remove_from_list(Parser(query).columns, 'NUM'))

            df['statement_with_cls'] = df[['statement', 'tables', 'attributes']].apply(
                lambda x: insert_cls_tokens(x['statement'], x['tables'], x['attributes']), 
                axis=1
            )
            
            df.to_csv(with_cls_data_dir_path + data_file.split('/')[-1])

        except:
            files_with_error = files_with_error + [data_file.split('/')[-1]]
    
    return files_with_error



with_cls_data_dir_path = create_dir_if_necessary(CONSTANTS.DATA_DIR_TRAIN_WITH_CLS)
raw_data_dir_path = get_absolute_path(CONSTANTS.DATA_DIR_TRAIN_RAW)
raw_data_files_list = []

for x in os.listdir(raw_data_dir_path):
    if x.endswith(".csv"):
        raw_data_files_list.append(os.path.join(raw_data_dir_path, x))

files_with_error = write_files_with_cls_tokens(raw_data_files_list)

print("Train Data with CLS Tokens - First Step Done!")
print("Files with errors:")
print(files_with_error)


# Retry
raw_data_files_list = []

for name in files_with_error:
    raw_data_files_list.append(os.path.join(raw_data_dir_path, name))

files_with_error = write_files_with_cls_tokens(raw_data_files_list)

print("Train Data with CLS Tokens - Retry Step Done!")
print("Files with errors:")
print(files_with_error)

# Files with error:
# ['12170991.csv', '11717303.csv', '14665519.csv', '6763013.csv', '14665754.csv', '4401050.csv', '16504263.csv', '12170992.csv', '10785176.csv', '5379884.csv', '15898023.csv', '5397030.csv', '1422728.csv', '2394504.csv', '14665509.csv', '8334813.csv', '10134313.csv', '13237302.csv', '10785272.csv', '8794874.csv', '5970842.csv', '4597647.csv', '940444.csv', '9440138.csv', '414021.csv', '5784695.csv', '13226029.csv', '1713323.csv', '5420265.csv', '9727996.csv', '8777032.csv', '10785177.csv', '60356.csv', '15796817.csv', '2646907.csv', '15879774.csv', '4728843.csv', '8619749.csv', '14665522.csv', '1704781.csv', '13740959.csv', '15966093.csv', '12171110.csv', '14665521.csv', '9419700.csv', '10986518.csv', '9422983.csv', '3454343.csv', '462863.csv', '10972517.csv', '14630580.csv', '122479.csv', '11932990.csv', '9485900.csv', '14313521.csv', '11036288.csv', '2399658.csv', '4285374.csv', '9110637.csv', '15029763.csv', '5970853.csv', '266006.csv', '3081825.csv', '11035307.csv', '16580943.csv', '7861715.csv', '13792630.csv', '16443293.csv', '4118031.csv', '233297.csv', '13740929.csv', '8520425.csv', '11700558.csv', '1481275.csv', '8380654.csv', '1546098.csv', '16445159.csv', '8677337.csv', '3714997.csv', '7531715.csv', '2785842.csv', '11401291.csv', '3030985.csv', '1803731.csv', '503574.csv', '14665512.csv', '8386053.csv', '9139629.csv', '3413773.csv', '4598476.csv', '5245286.csv', '1677705.csv', '9289496.csv', '3049888.csv', '2087074.csv', '10417791.csv', '9008353.csv', '9422984.csv', '13104476.csv', '7284168.csv', '10124158.csv', '11509959.csv', '13092984.csv', '10567485.csv', '14263440.csv', '2056539.csv', '10124159.csv', '4473976.csv', '16530250.csv', '7746938.csv', '1317802.csv', '5351274.csv', '12170712.csv', '2087052.csv', '16051106.csv', '5511441.csv', '11601721.csv', '3846317.csv', '1463927.csv', '10336083.csv', '6856320.csv', '12820429.csv', '11035291.csv', '8794885.csv', '2399655.csv', '6288419.csv', '7933263.csv', '2516878.csv', '503575.csv', '14665511.csv', '4544312.csv', '11735486.csv', '10937785.csv', '10124162.csv', '1793414.csv', '9472939.csv', '8248049.csv', '710453.csv', '3413772.csv', '314938.csv', '10378990.csv', '2087079.csv', '8677301.csv', '6755188.csv', '1422730.csv', '8619751.csv', '12158809.csv', '15966080.csv', '16059306.csv', '5420267.csv', '5552540.csv', '13827786.csv', '9110640.csv', '8404677.csv', '16059305.csv', '1458150.csv', '1716817.csv', '16530247.csv', '7719456.csv', '14665508.csv', '687068.csv', '6390678.csv', '8794872.csv', '13549122.csv', '43538.csv', '14490844.csv', '8186463.csv', '7922322.csv', '9870170.csv', '2785837.csv', '9380460.csv', '4691384.csv', '5157827.csv', '10785273.csv', '12089985.csv', '12089987.csv', '5970855.csv', '11525058.csv', '15282887.csv', '5007033.csv', '7001701.csv', '14665523.csv', '12170997.csv', '6552287.csv', '2827790.csv', '5821584.csv', '3528057.csv', '3859190.csv', '14086263.csv', '14010543.csv', '60388.csv', '1422734.csv', '14718755.csv', '13781554.csv', '9801125.csv', '14745954.csv', '14108926.csv', '9032841.csv', '3179163.csv', '2446297.csv', '1426636.csv', '12170812.csv', '464718.csv', '8453581.csv', '3773395.csv', '6924033.csv', '3435804.csv', '3474479.csv', '2711780.csv', '3063815.csv', '14551534.csv', '3538988.csv', '13808918.csv', '5733998.csv', '9020582.csv', '67390.csv', '6940588.csv', '2516884.csv', '268491.csv', '8703705.csv', '4830206.csv', '9106525.csv', '4831183.csv', '14022850.csv', '2773827.csv', '3070185.csv', '2655872.csv', '15148604.csv', '3268742.csv', '14745762.csv', '5721082.csv', '5841955.csv', '5016480.csv', '2797523.csv', '6732838.csv', '491643.csv', '2736317.csv', '15835765.csv', '15465293.csv', '10393670.csv', '8901904.csv', '8172882.csv', '6594575.csv', '9358888.csv', '12136189.csv', '7326979.csv', '2472933.csv', '39376.csv', '14948871.csv', '4947341.csv', '550752.csv', '6121041.csv', '5524910.csv', '5309166.csv', '8333538.csv', '2036144.csv', '11968789.csv', '14078744.csv', '6357608.csv', '4971843.csv', '810208.csv', '12008227.csv', '6361825.csv', '2767909.csv', '6065304.csv', '845916.csv', '7614258.csv', '7016429.csv', '16563037.csv', '564574.csv', '7346462.csv', '293409.csv', '7039109.csv', '15394983.csv', '3106656.csv', '14554763.csv', '10043942.csv', '3586048.csv', '3281730.csv', '485062.csv', '5905568.csv', '4927891.csv', '14718764.csv', '2279561.csv', '12617696.csv', '6216936.csv', '978663.csv', '6884159.csv', '11922753.csv', '1418290.csv', '8677304.csv', '11365378.csv', '13990002.csv', '15359213.csv', '6111352.csv', '8544727.csv', '12101033.csv', '942268.csv', '13924645.csv', '5178940.csv', '14440992.csv', '13081466.csv', '15450999.csv', '11409286.csv', '4354872.csv', '9988968.csv', '15615339.csv', '2958454.csv', '2012742.csv', '14017139.csv', '3167607.csv', '4090123.csv', '4425808.csv', '5598947.csv', '16019328.csv', '3589350.csv', '12208448.csv', '13840289.csv', '11582645.csv', '49861.csv', '13696181.csv', '15961428.csv', '15466253.csv', '7148873.csv', '9076577.csv', '13436171.csv', '3681447.csv', '12778576.csv', '15148543.csv', '10265172.csv', '4445126.csv', '13142499.csv', '11847839.csv', '3901639.csv', '4170574.csv', '13034215.csv', '4327436.csv', '15908129.csv', '9838403.csv', '15493233.csv', '6324773.csv', '1716820.csv', '9929089.csv', '2516907.csv', '3362277.csv', '7055434.csv', '6941625.csv', '2051766.csv', '16339814.csv', '14746337.csv', '9723760.csv', '13781738.csv', '14535972.csv', '13523849.csv', '14956227.csv', '14968274.csv', '12171601.csv', '8317784.csv', '8286501.csv', '12224251.csv', '16072959.csv', '8455895.csv', '2458747.csv', '14746137.csv', '16513592.csv', '10159193.csv', '15109449.csv', '6780447.csv', '3106655.csv', '6521850.csv', '12871586.csv', '6805569.csv', '12631037.csv', '1222534.csv', '11922755.csv', '1222565.csv', '3588307.csv', '16516919.csv', '9952223.csv', '11497398.csv', '8334812.csv', '10216750.csv', '13882171.csv', '6754398.csv', '891896.csv', '11594327.csv', '8677353.csv', '1863491.csv', '6084007.csv', '15394716.csv', '1692474.csv', '1926506.csv', '5261242.csv', '12170498.csv', '16563043.csv', '3106651.csv', '16181075.csv', '10378991.csv', '14727633.csv', '15216866.csv', '10072751.csv', '1107123.csv', '5764609.csv', '12539722.csv', '5126774.csv', '6404371.csv', '4445125.csv', '15022276.csv', '14143480.csv', '13888490.csv', '11982944.csv', '10336053.csv', '964645.csv', '1888906.csv', '12170792.csv', '14143469.csv', '12282402.csv', '2921092.csv', '13755056.csv', '4169950.csv', '2247781.csv', '10184095.csv', '12922605.csv', '9440096.csv', '6556899.csv', '7773962.csv', '12171523.csv', '8645220.csv', '13882146.csv', '3435803.csv', '13060228.csv', '3894546.csv', '6463529.csv', '15359995.csv', '14746141.csv', '3362171.csv', '10306561.csv', '14095699.csv', '12001999.csv', '1060299.csv', '8558671.csv', '5098895.csv', '16019023.csv', '6473489.csv', '9746084.csv', '8574529.csv', '14745741.csv', '15747269.csv', '1635310.csv', '3179180.csv', '12170683.csv', '60373.csv', '12398456.csv', '5552350.csv', '2461542.csv', '43540.csv', '13034442.csv', '11401268.csv', '13137358.csv', '1618052.csv', '6169512.csv', '15080883.csv', '8961130.csv', '6822839.csv', '13119683.csv', '4797886.csv', '4612083.csv', '10709944.csv', '8713732.csv', '7402379.csv', '15133204.csv', '11438451.csv', '58569.csv', '15632601.csv', '2785856.csv', '12360047.csv', '12912869.csv', '2012919.csv', '11428843.csv', '15216817.csv', '294180.csv', '14092274.csv', '1431643.csv', '13523912.csv', '3106765.csv', '6518195.csv', '7701575.csv', '9437962.csv', '8248047.csv', '2036154.csv', '13835710.csv', '1237239.csv', '13280615.csv', '11245329.csv', '10345407.csv', '7260967.csv', '10864658.csv', '270333.csv', '2012914.csv', '11279417.csv', '13090888.csv', '4932276.csv', '4981659.csv', '15216831.csv', '5424958.csv', '3106899.csv', '7624801.csv', '4132507.csv', '6555469.csv', '4635900.csv', '1040339.csv', '4184370.csv', '6426423.csv', '3687075.csv', '4403632.csv', '13088229.csv', '15777430.csv', '14017107.csv', '7166668.csv', '2546245.csv', '606358.csv', '4285917.csv', '11688985.csv', '5906465.csv', '11003615.csv', '14730734.csv', '12724023.csv', '1415668.csv', '12065775.csv', '13740926.csv', '13263855.csv', '2261997.csv', '8593739.csv', '1018136.csv', '10037800.csv', '7910576.csv', '14945456.csv', '8110664.csv', '15216856.csv', '729836.csv', '11949908.csv', '14660444.csv', '12416319.csv', '13523879.csv', '4726512.csv', '2087072.csv', '16240965.csv', '2777186.csv', '15216857.csv', '4961977.csv', '8252882.csv', '3499313.csv', '5389446.csv', '16024360.csv', '268487.csv', '6303927.csv', '9419754.csv', '4572389.csv', '5953506.csv', '7316760.csv', '1210281.csv', '13320723.csv', '12171711.csv', '8991929.csv', '6297309.csv', '12705111.csv', '15369921.csv', '1246555.csv', '12499755.csv', '7054825.csv', '406984.csv', '9641540.csv', '11244266.csv', '3063976.csv', '10838713.csv', '10751350.csv', '4691388.csv', '9615578.csv', '3291492.csv', '4932275.csv', '14746347.csv', '15933871.csv', '8014285.csv', '13872077.csv', '12745604.csv', '4808411.csv', '13800256.csv', '16341722.csv', '3106908.csv', '3499362.csv', '12751534.csv', '60361.csv', '9974270.csv', '11497397.csv', '12171162.csv', '16455544.csv', '14625774.csv', '170218.csv', '2868252.csv', '16051269.csv', '12171226.csv', '5295817.csv', '15864807.csv', '8404681.csv', '778406.csv', '13876380.csv', '3818319.csv', '7837260.csv', '1795293.csv', '1207287.csv', '13111790.csv', '3859177.csv', '2370219.csv', '4703680.csv', '14948834.csv', '5016464.csv', '1674186.csv', '14155203.csv', '15148463.csv', '2276331.csv', '16329146.csv', '4750172.csv', '10383355.csv', '14948956.csv', '3461307.csv', '8520452.csv', '2012750.csv', '6058970.csv', '4765026.csv']