
import pickle
import os
import numpy as np


def load_data():
    cifar_mean = np.array([0.5125891, 0.5335556, 0.5198208, 0.51035565, 0.5311504, 0.51707786, 0.51392424, 0.5343016, 0.5199328, 0.51595825, 0.535995, 0.5210931, 0.51837546, 0.5381541, 0.5226226, 0.5209901, 0.5406102, 0.52463686, 0.52302873, 0.5422941, 0.526108, 0.5250642, 0.54408634, 0.52755016, 0.5273931, 0.5461816, 0.5294758, 0.52915096, 0.5475848, 0.53056467, 0.53033257, 0.5487945, 0.53142065, 0.53100014, 0.5493823, 0.532008, 0.5318505, 0.5500815, 0.53262335, 0.53190315, 0.550054, 0.53259194, 0.53190464, 0.5499133, 0.53247046, 0.53181446, 0.54970396, 0.5322407, 0.53149545, 0.54939365, 0.53170216, 0.53139186, 0.5492686, 0.5315995, 0.53134245, 0.5492373, 0.5316772, 0.53092, 0.54887486, 0.5312779, 0.52984685, 0.54790294, 0.53024596, 0.5292543, 0.5475318, 0.53011733, 0.5286725, 0.5470605, 0.5298459, 0.52755964, 0.5462715, 0.5291427, 0.52567685, 0.54489666, 0.52795166, 0.52395415, 0.5434869, 0.52681994, 0.5218709, 0.5416036, 0.52525735, 0.51892465, 0.5388906, 0.52305186, 0.5160373, 0.5363746, 0.5209598, 0.51396006, 0.5345421, 0.5193824, 0.51134723, 0.532196, 0.517654, 0.5104877, 0.53127444, 0.51722926, 0.5101904, 0.5305694, 0.51515365, 0.50723326, 0.52729446, 0.5116333, 0.51065314, 0.53010774, 0.5141287, 0.5120484, 0.5311018, 0.5145084, 0.51433307, 0.5330152, 0.5158267, 0.5169507, 0.5352452, 0.5177454, 0.5190629, 0.5368237, 0.5191274, 0.52094585, 0.5383659, 0.52044463, 0.5231099, 0.5401586, 0.5219691, 0.5250054, 0.54164994, 0.5231126, 0.5265287, 0.5431024, 0.5243246, 0.527122, 0.54356045, 0.52482873, 0.52799946, 0.54437166, 0.5255006, 0.5280943, 0.5443916, 0.5255538, 0.52800775, 0.5441322, 0.52538264, 0.5272869, 0.54332507, 0.52461404, 0.52708703, 0.5431628, 0.5242423, 0.5274033, 0.5433042, 0.52435017, 0.5273274, 0.5431927, 0.5242712, 0.5270461, 0.5430902, 0.52421975, 0.5260983, 0.5423093, 0.52339584, 0.52580154, 0.54231465, 0.52348256, 0.52505565, 0.54183424, 0.5231322, 0.5237481, 0.5409074, 0.522347, 0.5214467, 0.5391383, 0.520837, 0.5197676, 0.5377843, 0.51964325, 0.5174169, 0.5357008, 0.5177166, 0.51449, 0.5332585, 0.51589715, 0.5119284, 0.53136545, 0.5144992, 0.50997263, 0.52995944, 0.5133403, 0.50717854, 0.52768564, 0.5117646, 0.5064974, 0.526998, 0.5115747, 0.5087227, 0.52803206, 0.5107892, 0.5047709, 0.5235213, 0.50595963, 0.50773895, 0.5257103, 0.5077519, 0.50922745, 0.5265793, 0.5080623, 0.5112224, 0.5278405, 0.5087765, 0.51357174, 0.5294538, 0.5100967, 0.51561373, 0.53096634, 0.5113942, 0.5176565, 0.5324904, 0.51281357, 0.5202049, 0.53453314, 0.51450443, 0.5220908, 0.53599066, 0.51567423, 0.523728, 0.5374099, 0.5169516, 0.5246113, 0.53810054, 0.5176903, 0.5249225, 0.53840846, 0.51783085, 0.5250187, 0.5384936, 0.5179714, 0.5246734, 0.53794736, 0.51762265, 0.52400243, 0.5370531, 0.5168342, 0.5241845, 0.5371431, 0.51674104, 0.52449924, 0.5372522, 0.51672924, 0.5242473, 0.53696275, 0.51642936, 0.5235294, 0.5365606, 0.5161019, 0.52258, 0.53588074, 0.5154991, 0.5221798, 0.535868, 0.5153978, 0.52158374, 0.53566223, 0.5151015, 0.52009076, 0.5346052, 0.51411974, 0.5178056, 0.5329729, 0.5127169, 0.5159323, 0.53150475, 0.51149535, 0.5138075, 0.52980405, 0.51005995, 0.5114662, 0.5282267, 0.50894624, 0.50914806, 0.52683425, 0.5080967, 0.50671124, 0.5253311, 0.50703907, 0.50428075, 0.52369565, 0.5061419, 0.50427943, 0.52375823, 0.50676244, 0.50769454, 0.5258301, 0.50687677, 0.50341964, 0.5208474, 0.50148106, 0.50609, 0.5224348, 0.5025353, 0.50786495, 0.5232852, 0.50283676, 0.5093051, 0.52371645, 0.502746, 0.51104456, 0.5243688, 0.5031271, 0.5130025, 0.5254102, 0.50404936, 0.51548815, 0.52718323, 0.50571525, 0.5182901, 0.5295068, 0.5077093, 0.5202635, 0.53101903, 0.50896144, 0.52161074, 0.5320202, 0.5097878, 0.5223921, 0.5325996, 0.51047987, 0.5221468, 0.5323063, 0.51019615, 0.52245516, 0.5325286, 0.5104237, 0.522466, 0.5323604, 0.51043147, 0.5216972, 0.531244, 0.50954396, 0.5217702, 0.5311202, 0.5091887, 0.52198994, 0.531162, 0.50911826, 0.52152467, 0.5307484, 0.50871134, 0.5208862, 0.53046817, 0.50849295, 0.51973104, 0.52963537, 0.5076733, 0.518899, 0.52925825, 0.5072149, 0.5185742, 0.5293947, 0.5070613, 0.5171974, 0.5285756, 0.5062711, 0.5151835, 0.52734274, 0.5052106, 0.5133427, 0.526205, 0.5043163, 0.51124775, 0.52483165, 0.5032927, 0.5095077, 0.52403444, 0.5028577, 0.50765795, 0.52329034, 0.50265884, 0.5052181, 0.52209514, 0.5019957, 0.50280863, 0.52072275, 0.50135434, 0.50321615, 0.52139026, 0.5026594, 0.5066728, 0.52361757, 0.5029942, 0.5021932, 0.51821595, 0.49708524, 0.50442374, 0.51916325, 0.49741644, 0.5060885, 0.51950955, 0.49727476, 0.5069638, 0.519059, 0.4964266, 0.5087284, 0.51950014, 0.49678195, 0.51071334, 0.5201907, 0.49735886, 0.51288563, 0.5213807, 0.49839514, 0.51536316, 0.5230849, 0.49987316, 0.51695275, 0.5241572, 0.50066316, 0.5187275, 0.52559185, 0.50181437, 0.5197409, 0.5263634, 0.5026216, 0.5193163, 0.5257978, 0.5021233, 0.5194247, 0.5257926, 0.50220615, 0.51956624, 0.52580345, 0.50242853, 0.5187764, 0.52467954, 0.5014981, 0.51859564, 0.524318, 0.5010181, 0.5188952, 0.5245507, 0.5012477, 0.5188869, 0.52466905, 0.50141686, 0.5179967, 0.52408093, 0.5008127, 0.51634544, 0.522779, 0.49937427, 0.51529515, 0.5221451, 0.49863207, 0.5147205, 0.5221899, 0.49849948, 0.5136438, 0.5219193, 0.49821168, 0.512231, 0.52134496, 0.49777523, 0.51080537, 0.5208963, 0.49745238, 0.5089362, 0.52011764, 0.49691004, 0.50721794, 0.5195741, 0.49661744, 0.5056025, 0.51918864, 0.49675563, 0.5035451, 0.51855266, 0.49662897, 0.50160587, 0.51794636, 0.49665856, 0.5021448, 0.5190644, 0.49849382, 0.50551444, 0.5212539, 0.49905226, 0.500901, 0.515494, 0.49274132, 0.50242263, 0.5154062, 0.49202266, 0.50367904, 0.51508915, 0.4912601, 0.50437975, 0.5142034, 0.49006906, 0.5060234, 0.5143099, 0.49016085, 0.5077508, 0.51451904, 0.4902386, 0.50932765, 0.51468265, 0.4902106, 0.51161623, 0.51585156, 0.49129432, 0.51345116, 0.51689655, 0.4920814, 0.5153487, 0.5183447, 0.49324933, 0.51650304, 0.5192187, 0.49404824, 0.5158884, 0.51828694, 0.49310726, 0.5160163, 0.51822007, 0.4932475, 0.5161736, 0.5182637, 0.49356017, 0.51546353, 0.51731044, 0.4927994, 0.5151736, 0.5170156, 0.4923346, 0.5157531, 0.517689, 0.49301156, 0.51606673, 0.5180764, 0.49351144, 0.5152466, 0.5174159, 0.4927852, 0.5140167, 0.5166419, 0.49183553, 0.51290643, 0.5161224, 0.49113527, 0.5120536, 0.51620877, 0.49119917, 0.5106604, 0.51582474, 0.49097508, 0.5096143, 0.5157343, 0.49093577, 0.5088257, 0.51619846, 0.49136695, 0.5070477, 0.51582706, 0.49112934, 0.5051287, 0.51543283, 0.4907929, 0.5036252, 0.51533824, 0.49112743, 0.5017869, 0.5151173, 0.49134406, 0.5001115, 0.51498383, 0.49185488, 0.50085896, 0.51651156, 0.49407417, 0.5038685, 0.51837873, 0.49433786, 0.49883226, 0.5121282, 0.48754528, 0.5000758, 0.5114122, 0.4863624, 0.5008112, 0.5102833, 0.4848477, 0.5017778, 0.50946885, 0.48368028, 0.50302577, 0.5089437, 0.48305368, 0.50465435, 0.50862306, 0.48278242, 0.5060686, 0.50833166, 0.4824641, 0.5082394, 0.5090778, 0.48314664, 0.51039714, 0.50986594, 0.48389146, 0.5121611, 0.51069266, 0.484579, 0.51295584, 0.51114124, 0.4849066, 0.5123309, 0.51033, 0.48397908, 0.51237476, 0.51005167, 0.48390156, 0.5126856, 0.5102092, 0.48433435, 0.51228243, 0.50971967, 0.48407215, 0.5118439, 0.5095021, 0.4836986, 0.51235306, 0.51017654, 0.48434585, 0.5126779, 0.5105838, 0.4847964, 0.5120654, 0.5102172, 0.48434183, 0.51134646, 0.51007485, 0.483996, 0.510205, 0.50959635, 0.4832043, 0.5089302, 0.5094197, 0.48296732, 0.50774676, 0.5095382, 0.48338902, 0.50677013, 0.5098224, 0.48369464, 0.50615335, 0.51083356, 0.48462817, 0.50475997, 0.5111398, 0.48498636, 0.5028428, 0.51094466, 0.48488927, 0.5013838, 0.5112705, 0.4854046, 0.49967837, 0.5114891, 0.48588443, 0.4981652, 0.511602, 0.48674685, 0.4992814, 0.5136052, 0.4894055, 0.50194436, 0.51530254, 0.4893368, 0.49646285, 0.50839293, 0.48204112, 0.49751297, 0.50725454, 0.48049524, 0.4980085, 0.50569284, 0.47851276, 0.49864239, 0.5044038, 0.4768957, 0.5003046, 0.5039402, 0.47636697, 0.5018272, 0.50317806, 0.4758802, 0.5025877, 0.5020061, 0.47498694, 0.5044948, 0.5020804, 0.4750632, 0.50692207, 0.50274223, 0.47567427, 0.50856435, 0.5031034, 0.47595063, 0.50887084, 0.5028764, 0.47573155, 0.50831467, 0.50201786, 0.474696, 0.50878197, 0.5020216, 0.47488257, 0.5092103, 0.5021663, 0.4753759, 0.5088528, 0.501753, 0.47515774, 0.5090174, 0.5022259, 0.47556934, 0.50975955, 0.50308096, 0.47625992, 0.5098739, 0.50340456, 0.47644824, 0.50900525, 0.5029853, 0.4759902, 0.50806814, 0.50276726, 0.47558382, 0.5074465, 0.50283927, 0.47528836, 0.5062712, 0.5029108, 0.4752889, 0.50548434, 0.5036701, 0.4762001, 0.50404775, 0.5039275, 0.4764725, 0.50313514, 0.5049122, 0.47745422, 0.5022928, 0.5060837, 0.47858214, 0.50063115, 0.50638074, 0.47893777, 0.49917746, 0.50693405, 0.47952414, 0.49736926, 0.50737053, 0.48005804, 0.49621025, 0.5079847, 0.48137575, 0.4975825, 0.51045513, 0.48442462, 0.49982464, 0.51189977, 0.48419452, 0.49396628, 0.5043872, 0.47629082, 0.49500015, 0.50310415, 0.47462815, 0.4952585, 0.50118107, 0.47242218, 0.49591368, 0.49979934, 0.4707737, 0.49758473, 0.49904215, 0.46989417, 0.49899518, 0.49794295, 0.46906137, 0.49925107, 0.49597722, 0.46748066, 0.5008285, 0.49532843, 0.4670364, 0.50324535, 0.49585804, 0.467465, 0.50472397, 0.49579772, 0.46731308, 0.5053876, 0.49565825, 0.46742582, 0.5056196, 0.49522242, 0.4670161, 0.5063051, 0.4951073, 0.46714735, 0.5068745, 0.49522048, 0.46752492, 0.5073496, 0.49560735, 0.46799144, 0.507885, 0.49643242, 0.4688345, 0.50824773, 0.49709204, 0.46944156, 0.5082332, 0.4975203, 0.4696642, 0.5075482, 0.49744686, 0.4694545, 0.5059482, 0.4967136, 0.46864048, 0.50503707, 0.49667683, 0.46843484, 0.5044864, 0.4975461, 0.46908566, 0.50388604, 0.49857575, 0.46996602, 0.5026681, 0.49931762, 0.4705422, 0.50154334, 0.50041497, 0.47160476, 0.5003431, 0.5015314, 0.47257248, 0.49897608, 0.5024307, 0.4734876, 0.4971384, 0.50264883, 0.47366676, 0.49519473, 0.50315595, 0.47422406, 0.49401098, 0.50414294, 0.47581944, 0.4957519, 0.5071924, 0.47934458, 0.49893367, 0.5095179, 0.48008975, 0.4928944, 0.5017452, 0.4717997, 0.49338293, 0.49989566, 0.4696745, 0.49354276, 0.4976462, 0.46727362, 0.49438575, 0.49627277, 0.46561894, 0.49573293, 0.49502394, 0.46427327, 0.49726063, 0.49377844, 0.4633081, 0.49748582, 0.49128968, 0.46113962, 0.4989005, 0.49013296, 0.46027887, 0.5010044, 0.49008757, 0.46031466, 0.5025262, 0.48988083, 0.46005243, 0.50376666, 0.48998192, 0.46057153, 0.5048968, 0.49004027, 0.46088323, 0.50567216, 0.4897594, 0.4608909, 0.50619817, 0.48958594, 0.4609467, 0.5066509, 0.4898532, 0.46136156, 0.50730014, 0.49089676, 0.462447, 0.50729716, 0.49139687, 0.4630397, 0.50767636, 0.4923075, 0.4638307, 0.50731945, 0.49261403, 0.46379933, 0.505517, 0.49204, 0.46285215, 0.5039381, 0.4918244, 0.4623946, 0.50265783, 0.4921657, 0.46253702, 0.50201833, 0.49335298, 0.46347064, 0.5012656, 0.49502248, 0.4648494, 0.500538, 0.4967403, 0.46637148, 0.49912858, 0.49781212, 0.46717298, 0.4977372, 0.49885482, 0.46822107, 0.49562797, 0.49895254, 0.46833354, 0.49358562, 0.49967587, 0.46917835, 0.49239615, 0.5010549, 0.47097722, 0.49459463, 0.504631, 0.475064, 0.4979264, 0.50716776, 0.4759557, 0.4918032, 0.49916625, 0.46748352, 0.49213278, 0.49701986, 0.46518925, 0.492414, 0.49463385, 0.4625616, 0.4932275, 0.49297503, 0.46055982, 0.49494243, 0.49193472, 0.45942247, 0.49634242, 0.4903976, 0.45815954, 0.49722457, 0.4882345, 0.4563171, 0.49828753, 0.48629057, 0.45481145, 0.49942532, 0.48483828, 0.45377147, 0.5017263, 0.48501387, 0.45419106, 0.50325686, 0.4850058, 0.4545724, 0.5039892, 0.48445266, 0.4544924, 0.5051761, 0.48443985, 0.45475495, 0.5060425, 0.48451993, 0.45506874, 0.50670314, 0.4848033, 0.4554729, 0.5068166, 0.48527336, 0.45600176, 0.5070288, 0.48594874, 0.45693257, 0.5075, 0.48708707, 0.4578876, 0.5069521, 0.4875365, 0.45782277, 0.5054159, 0.48749712, 0.45726743, 0.5035466, 0.4874107, 0.4567758, 0.5017038, 0.4874609, 0.45657945, 0.50083023, 0.48875546, 0.45755926, 0.49977908, 0.49037874, 0.45895723, 0.49890742, 0.49226987, 0.46070153, 0.49780706, 0.4937254, 0.46175405, 0.4959546, 0.494491, 0.4624494, 0.4938109, 0.49503827, 0.46304774, 0.49181435, 0.49606878, 0.4641168, 0.49075583, 0.4977087, 0.46593586, 0.493292, 0.5019984, 0.47069708, 0.4967748, 0.50465685, 0.4717481, 0.49010172, 0.49596816, 0.4626404, 0.49042368, 0.49353567, 0.46001673, 0.49080706, 0.49099243, 0.4572523, 0.4921259, 0.4895786, 0.45560113, 0.49457067, 0.4889396, 0.45486704, 0.4959915, 0.4872486, 0.45347446, 0.49660495, 0.4847584, 0.45158485, 0.49721727, 0.48199192, 0.4493901, 0.49842682, 0.48014995, 0.4480981, 0.5010955, 0.48024452, 0.44856983, 0.5029377, 0.4802045, 0.44899738, 0.5042913, 0.48013192, 0.44936687, 0.5058132, 0.48020938, 0.4498175, 0.5067236, 0.4802015, 0.44999287, 0.5077323, 0.48068598, 0.45060402, 0.5077365, 0.48096684, 0.45091125, 0.5077012, 0.48149258, 0.4515584, 0.5078106, 0.4825217, 0.4524005, 0.506605, 0.48267448, 0.45216075, 0.50480986, 0.48264015, 0.45152882, 0.50299495, 0.48267096, 0.45105693, 0.50172853, 0.48341462, 0.45151478, 0.50032693, 0.48467547, 0.45240417, 0.49895766, 0.4861675, 0.4536566, 0.4976245, 0.4877157, 0.4549862, 0.49636078, 0.4892411, 0.4560303, 0.49428403, 0.49022096, 0.45697924, 0.49180847, 0.4909277, 0.45778778, 0.48999724, 0.49230546, 0.45898786, 0.4893065, 0.494253, 0.46093097, 0.49204266, 0.4991796, 0.46630606, 0.49550968, 0.50181705, 0.46721423, 0.48863742, 0.49280503, 0.4579277, 0.48859853, 0.4897394, 0.45469096, 0.48882475, 0.48680142, 0.4516573, 0.49046448, 0.48538777, 0.4501436, 0.4935391, 0.48516476, 0.44997358, 0.4955372, 0.48386702, 0.44903612, 0.49636477, 0.48131943, 0.44729123, 0.49690264, 0.47835723, 0.44494087, 0.49877423, 0.47686034, 0.4438899, 0.50176084, 0.4768353, 0.44408223, 0.5042957, 0.477343, 0.44499835, 0.50582796, 0.47715458, 0.44520703, 0.5071629, 0.47681445, 0.44540218, 0.5081276, 0.47671828, 0.44564548, 0.50898445, 0.4769926, 0.44604456, 0.50934464, 0.47755072, 0.44655782, 0.50911057, 0.4777294, 0.44670698, 0.50882787, 0.47842446, 0.4472453, 0.5075528, 0.47873765, 0.44718543, 0.5050821, 0.4781854, 0.4461337, 0.5033845, 0.47834134, 0.44580433, 0.5022064, 0.47943324, 0.44647598, 0.5007013, 0.4810909, 0.44763651, 0.49922243, 0.4827641, 0.44886023, 0.49722716, 0.4839076, 0.4496609, 0.49541944, 0.48530293, 0.45072067, 0.49316433, 0.48645008, 0.45187888, 0.49026492, 0.48719046, 0.4528157, 0.48836845, 0.48866087, 0.4540422, 0.4882927, 0.49136755, 0.45654684, 0.4910425, 0.49652937, 0.46207708, 0.49370152, 0.49863848, 0.4624133, 0.48657668, 0.48924345, 0.45284316, 0.4867835, 0.4861309, 0.44971207, 0.4876669, 0.4834556, 0.4470639, 0.48964882, 0.4820599, 0.44565332, 0.4928811, 0.48187423, 0.44568536, 0.49492255, 0.48032874, 0.44460174, 0.49619076, 0.4779162, 0.44288433, 0.49764556, 0.47569376, 0.4412587, 0.50030124, 0.4746849, 0.4408077, 0.5034509, 0.47453582, 0.44089746, 0.5063687, 0.47512567, 0.4417419, 0.5077552, 0.47457618, 0.4414889, 0.5086126, 0.47358114, 0.44109887, 0.50960577, 0.4734714, 0.4413365, 0.5101132, 0.47348964, 0.44145644, 0.5102535, 0.47361833, 0.44143867, 0.5103364, 0.47384787, 0.4415993, 0.5099839, 0.4744189, 0.44202864, 0.5087035, 0.4747229, 0.44187722, 0.5069964, 0.47504497, 0.44177803, 0.5050453, 0.47550285, 0.44180676, 0.5029002, 0.47600633, 0.44186854, 0.5012443, 0.4777499, 0.44327393, 0.49890348, 0.47885805, 0.44391027, 0.49647012, 0.47983184, 0.44441485, 0.4944723, 0.48138154, 0.44565898, 0.49129403, 0.48196852, 0.4461107, 0.4883029, 0.48287582, 0.44705695, 0.48648065, 0.48472503, 0.44855005, 0.48675364, 0.48811013, 0.4517, 0.4898694, 0.49387822, 0.4576936, 0.49267325, 0.4965016, 0.45861888, 0.48472813, 0.48607555, 0.44821176, 0.4851703, 0.4828152, 0.44512334, 0.48680007, 0.48050714, 0.44295296, 0.48943, 0.47946402, 0.44205004, 0.4921082, 0.47849604, 0.44130418, 0.49403518, 0.47665903, 0.43996602, 0.49607074, 0.4748696, 0.43872422, 0.49874878, 0.47363934, 0.4379633, 0.50116795, 0.47220817, 0.4371685, 0.5036451, 0.4712333, 0.43659544, 0.5060372, 0.4710541, 0.43675196, 0.5080405, 0.47097656, 0.436905, 0.50906533, 0.4701898, 0.4366737, 0.5095589, 0.46971977, 0.43645054, 0.5095605, 0.46912843, 0.43589434, 0.5096588, 0.46888188, 0.43547544, 0.50995463, 0.46909863, 0.43539128, 0.50998944, 0.47021738, 0.43626815, 0.50925773, 0.4711538, 0.43680277, 0.50770724, 0.47178468, 0.4370732, 0.50496185, 0.47179237, 0.43667352, 0.50200653, 0.4718798, 0.43640015, 0.4993973, 0.47282925, 0.43727824, 0.49696958, 0.47378168, 0.43797952, 0.49442086, 0.47489202, 0.43858007, 0.49180892, 0.4760893, 0.4394226, 0.48863047, 0.4769292, 0.43997362, 0.48606238, 0.478356, 0.4412558, 0.48466715, 0.48089924, 0.44332278, 0.4849401, 0.4846041, 0.44666097, 0.48882532, 0.4914228, 0.4535407, 0.49268207, 0.4952098, 0.45585966, 0.48433807, 0.4841154, 0.4449786, 0.48421884, 0.48015177, 0.44136867, 0.48533133, 0.47725466, 0.43870497, 0.48759267, 0.47559616, 0.43719527, 0.4899907, 0.47400004, 0.43588513, 0.49257335, 0.4728307, 0.4350741, 0.49549356, 0.471993, 0.43457165, 0.49732798, 0.46990857, 0.4329577, 0.4993391, 0.4679355, 0.43147355, 0.50131243, 0.46622264, 0.43016183, 0.5035003, 0.46563327, 0.43004262, 0.50586057, 0.46594933, 0.43063706, 0.50671315, 0.46505433, 0.43027684, 0.50720286, 0.4644783, 0.43007416, 0.5065652, 0.4632923, 0.42896625, 0.50674206, 0.463173, 0.42846307, 0.5072345, 0.4635861, 0.42852175, 0.50755554, 0.46496657, 0.4294866, 0.5070834, 0.46635282, 0.43042985, 0.50499576, 0.466583, 0.43030617, 0.5020161, 0.46619397, 0.42961118, 0.49960917, 0.46692374, 0.43010715, 0.49671757, 0.46757874, 0.43065038, 0.494029, 0.4684111, 0.43146124, 0.4915086, 0.46969151, 0.4324231, 0.48884016, 0.47106746, 0.43367115, 0.48577976, 0.47217187, 0.43445578, 0.48384765, 0.4743089, 0.4362553, 0.48255178, 0.47712544, 0.43854022, 0.4827859, 0.48104542, 0.4419203, 0.4875408, 0.4889175, 0.44980296, 0.49222857, 0.49362954, 0.4531146, 0.48345134, 0.48189873, 0.44176653, 0.48254034, 0.47699937, 0.4373796, 0.4828492, 0.47326526, 0.43395284, 0.48428833, 0.47077665, 0.43156943, 0.48657024, 0.46893388, 0.42990613, 0.4895016, 0.46800038, 0.42916775, 0.49225304, 0.4670462, 0.4283955, 0.4933782, 0.46426156, 0.42603403, 0.49568698, 0.46265113, 0.4248805, 0.49803782, 0.4613017, 0.4239555, 0.49997136, 0.46038866, 0.4236022, 0.50222784, 0.46056995, 0.42400566, 0.5032506, 0.45997098, 0.42384174, 0.50299764, 0.45844653, 0.4227519, 0.50191563, 0.4568102, 0.4212024, 0.50173944, 0.45661685, 0.4206811, 0.50181127, 0.45671672, 0.42050356, 0.5017804, 0.45769, 0.42103347, 0.50113153, 0.45878524, 0.42163467, 0.4995472, 0.45958441, 0.42210057, 0.49701414, 0.45966932, 0.42187333, 0.4948802, 0.46053046, 0.42259118, 0.49240312, 0.46150187, 0.42342836, 0.48942497, 0.46234503, 0.42430562, 0.4871664, 0.4639641, 0.42587113, 0.48441172, 0.4653442, 0.42727435, 0.48171046, 0.46682847, 0.42855245, 0.48050335, 0.46968222, 0.4309223, 0.47995222, 0.4733025, 0.4338385, 0.4811531, 0.47816625, 0.4381573, 0.4867816, 0.48707005, 0.44702503, 0.49141234, 0.49194038, 0.45037937, 0.48170608, 0.47924593, 0.43815848, 0.47971013, 0.47332907, 0.43277887, 0.47890407, 0.46855375, 0.42848718, 0.48002294, 0.4656702, 0.4257275, 0.48198983, 0.46358782, 0.4237998, 0.48515794, 0.46271095, 0.42309296, 0.48783034, 0.4617173, 0.42214933, 0.48901296, 0.45926648, 0.41988516, 0.49089366, 0.45746103, 0.41848654, 0.49287525, 0.45598665, 0.41764778, 0.49450523, 0.45486867, 0.4170744, 0.4964473, 0.45471716, 0.41693056, 0.4979029, 0.45449352, 0.41694745, 0.49695715, 0.45247337, 0.4152949, 0.49592242, 0.4508635, 0.4137969, 0.49496156, 0.44978628, 0.4124712, 0.494225, 0.44924206, 0.4118363, 0.4940134, 0.45002058, 0.4123555, 0.49361187, 0.4513441, 0.41334206, 0.49261862, 0.45272407, 0.41449594, 0.4910123, 0.4537939, 0.4151827, 0.4891517, 0.45477542, 0.4160118, 0.4870745, 0.45586804, 0.4169564, 0.48419368, 0.45671332, 0.4176718, 0.48176137, 0.45807993, 0.41909894, 0.47938812, 0.45961028, 0.42068803, 0.47727743, 0.4615559, 0.42247027, 0.47666398, 0.46509635, 0.42561924, 0.47699735, 0.4695791, 0.42939344, 0.47910586, 0.47532177, 0.43453637, 0.48533195, 0.48489404, 0.4440362, 0.4909336, 0.49069473, 0.44812757, 0.4800212, 0.47689426, 0.4350016, 0.47652468, 0.46948406, 0.4282753, 0.4748533, 0.46400884, 0.42318982, 0.4753451, 0.4606548, 0.41995788, 0.4770718, 0.45832232, 0.41774407, 0.47998485, 0.4573137, 0.41691777, 0.4826587, 0.45660785, 0.41644394, 0.48434055, 0.45490175, 0.4147834, 0.4857314, 0.45285335, 0.4131834, 0.48699182, 0.45115927, 0.41208753, 0.4885743, 0.45024708, 0.41164368, 0.48958632, 0.44931144, 0.41066742, 0.4908836, 0.44885212, 0.41009766, 0.490033, 0.4469305, 0.40853015, 0.48908585, 0.44562486, 0.40742284, 0.4880996, 0.4444442, 0.40593874, 0.48719195, 0.4437357, 0.40511867, 0.4868416, 0.44438827, 0.40560266, 0.48649922, 0.4457055, 0.40672186, 0.48552075, 0.4469248, 0.40770343, 0.48465618, 0.4485857, 0.4090899, 0.48288673, 0.44959667, 0.40996966, 0.48128062, 0.45089018, 0.41112146, 0.47870058, 0.45166087, 0.41180938, 0.4761865, 0.45259836, 0.41288528, 0.47404155, 0.4540715, 0.41436234, 0.47269, 0.45650285, 0.4165419, 0.47286126, 0.4606648, 0.42024902, 0.47428715, 0.46630406, 0.42518413, 0.47749433, 0.47324675, 0.4316184, 0.48468167, 0.4836629, 0.4418336, 0.49038738, 0.4894758, 0.44599137, 0.4786051, 0.4748186, 0.43217805, 0.4739541, 0.46643615, 0.424547, 0.47161707, 0.46041438, 0.41872394, 0.47088465, 0.45603442, 0.41448262, 0.47172025, 0.45304585, 0.4116466, 0.47422644, 0.45193648, 0.41083297, 0.47647485, 0.45104218, 0.4102996, 0.47871363, 0.45009074, 0.40941817, 0.47970176, 0.44806504, 0.40758622, 0.48077902, 0.44673073, 0.4066285, 0.48239556, 0.4461381, 0.406507, 0.48310184, 0.44495335, 0.40536308, 0.48341256, 0.44372135, 0.4040446, 0.4827836, 0.44205242, 0.40253726, 0.48210746, 0.44118226, 0.40183714, 0.4814093, 0.44043845, 0.40085673, 0.48094538, 0.44009113, 0.4003602, 0.4807507, 0.44092187, 0.40108305, 0.48013055, 0.44185352, 0.4018946, 0.47915024, 0.44284657, 0.402518, 0.4784955, 0.44449207, 0.40387926, 0.47708333, 0.44557968, 0.40490234, 0.47542992, 0.44651788, 0.405837, 0.47325695, 0.44724196, 0.40666974, 0.47111505, 0.4484366, 0.408051, 0.46916032, 0.44997087, 0.40946445, 0.4687677, 0.4528788, 0.4120764, 0.46996024, 0.4576584, 0.41642952, 0.47214544, 0.4639683, 0.42203125, 0.47641894, 0.4718391, 0.42949817, 0.48446226, 0.48280293, 0.44016317, 0.4904502, 0.4889342, 0.44461867, 0.47747833, 0.4732471, 0.4297045, 0.47209415, 0.46440262, 0.4217454, 0.46893883, 0.4577043, 0.41530457, 0.46747503, 0.4528644, 0.41063258, 0.46705288, 0.44910118, 0.40710685, 0.4689187, 0.4476753, 0.40596816, 0.4710917, 0.44706544, 0.4054308, 0.47322798, 0.44637996, 0.40480036, 0.47401327, 0.44458035, 0.40312898, 0.47440195, 0.44284812, 0.4015886, 0.4755172, 0.44188988, 0.40109208, 0.47648847, 0.44119617, 0.40067655, 0.47685918, 0.4402483, 0.39973855, 0.47609216, 0.4386831, 0.3982288, 0.47522417, 0.43767172, 0.39710727, 0.47519797, 0.43756765, 0.3967371, 0.47549462, 0.43796265, 0.39704254, 0.47474524, 0.43806058, 0.39721614, 0.474436, 0.43934363, 0.39846614, 0.4740963, 0.4407958, 0.39963016, 0.4731836, 0.44178876, 0.40022632, 0.47166526, 0.44230458, 0.40058738, 0.47036082, 0.4431542, 0.4014316, 0.46896273, 0.44433352, 0.4027738, 0.46788573, 0.44630224, 0.40493828, 0.46593282, 0.44764116, 0.40617445, 0.46581706, 0.45055774, 0.40885815, 0.467148, 0.4552901, 0.413343, 0.47015914, 0.46221632, 0.41955948, 0.47590438, 0.47117254, 0.42809808, 0.48453513, 0.48240784, 0.43900448, 0.4907998, 0.48875153, 0.44391593, 0.4766856, 0.47221425, 0.4280176, 0.47078082, 0.46314657, 0.41983324, 0.46716425, 0.45636868, 0.41335848, 0.46501, 0.45111534, 0.4082495, 0.46391618, 0.44701654, 0.4043772, 0.46490332, 0.44515824, 0.40265593, 0.4667201, 0.44454247, 0.40199926, 0.46827796, 0.44361395, 0.40106985, 0.4689617, 0.44204637, 0.39977044, 0.46984524, 0.44108343, 0.39902535, 0.47006163, 0.43934757, 0.39769718, 0.4703922, 0.43820643, 0.3969725, 0.47107235, 0.4376927, 0.39650938, 0.47022298, 0.43624225, 0.39514133, 0.46930453, 0.43529567, 0.39403427, 0.46982712, 0.43576032, 0.39404446, 0.47017223, 0.43618742, 0.39435542, 0.4694003, 0.4360398, 0.39428154, 0.46953455, 0.43755886, 0.3957795, 0.46984398, 0.43946698, 0.3975351, 0.46860978, 0.4398625, 0.39742172, 0.46743664, 0.44045395, 0.39771962, 0.4664153, 0.441422, 0.39881715, 0.46539003, 0.4425812, 0.4001243, 0.46471027, 0.44455647, 0.40233374, 0.4631384, 0.445892, 0.4036109, 0.46269837, 0.4483479, 0.40574276, 0.46464917, 0.45352122, 0.4106488, 0.46913794, 0.46151072, 0.4180183, 0.47612476, 0.47142535, 0.42751458, 0.48562384, 0.48328254, 0.4391903, 0.49065673, 0.48847106, 0.44337398, 0.4762089, 0.47174934, 0.42720953, 0.46992597, 0.46254256, 0.41864532, 0.46587697, 0.455665, 0.41211063, 0.4632615, 0.4503376, 0.40690297, 0.4615228, 0.44599396, 0.40270564, 0.46145, 0.44347313, 0.40029544, 0.46229553, 0.44214556, 0.39903614, 0.4633629, 0.4409681, 0.39787462, 0.46444836, 0.4400592, 0.3970369, 0.46541107, 0.4395973, 0.396772, 0.46545494, 0.4378673, 0.39552382, 0.46544418, 0.4364894, 0.3945714, 0.46567369, 0.4357634, 0.39408976, 0.46517593, 0.4347266, 0.3931963, 0.4646447, 0.4339637, 0.39231563, 0.46494365, 0.43425128, 0.3921338, 0.4652371, 0.4346358, 0.39233696, 0.46531063, 0.4351833, 0.39275497, 0.4658149, 0.4368641, 0.3942792, 0.46608558, 0.43857804, 0.39584464, 0.46512866, 0.439183, 0.3959698, 0.46426895, 0.43989575, 0.39645782, 0.4629032, 0.4403246, 0.3970551, 0.46206772, 0.44136095, 0.398182, 0.4615198, 0.4430804, 0.40006167, 0.46072856, 0.44482896, 0.4017565, 0.4604609, 0.44716954, 0.40379632, 0.4634732, 0.45312282, 0.40951848, 0.4690215, 0.4616619, 0.4175838, 0.4761963, 0.4715245, 0.42697492, 0.48598465, 0.48345333, 0.43864906, 0.49113593, 0.4888251, 0.44338593, 0.4773457, 0.47286874, 0.4278035, 0.47095615, 0.46379542, 0.41921082, 0.4659525, 0.4563613, 0.41217622, 0.4628005, 0.45087418, 0.40681058, 0.46057633, 0.44652542, 0.40242973, 0.4597378, 0.44349438, 0.39950204, 0.45995352, 0.44172272, 0.39771512, 0.46068993, 0.44055927, 0.39651066, 0.46182066, 0.43993324, 0.39588073, 0.46251452, 0.43930632, 0.39557284, 0.4624306, 0.43774733, 0.39479008, 0.46249196, 0.43669358, 0.3942852, 0.46214354, 0.43576548, 0.3936931, 0.46174237, 0.43486437, 0.39285848, 0.46176487, 0.43439063, 0.39235017, 0.4621464, 0.434697, 0.3922237, 0.4625178, 0.43519288, 0.39243484, 0.4627419, 0.43585622, 0.39291012, 0.4626646, 0.43701527, 0.3939037, 0.4623528, 0.4381942, 0.39481428, 0.46204403, 0.43909293, 0.39528152, 0.46148244, 0.439675, 0.39561906, 0.46060348, 0.4404073, 0.39633393, 0.45987886, 0.4413812, 0.39732507, 0.4594978, 0.442881, 0.39890796, 0.4589982, 0.44455284, 0.4007941, 0.4595094, 0.4473167, 0.40339577, 0.46308458, 0.45341235, 0.40925452, 0.4689954, 0.46212262, 0.41751015, 0.47654805, 0.47209722, 0.42697477, 0.4861872, 0.48372567, 0.43831775, 0.4921391, 0.4896993, 0.44395274, 0.4788669, 0.47454727, 0.42905304, 0.47187188, 0.46527085, 0.42010412, 0.46658456, 0.45792884, 0.41303116, 0.46274406, 0.45198298, 0.40729395, 0.46039143, 0.44771364, 0.40291768, 0.45922396, 0.4445714, 0.3996963, 0.45915, 0.4426975, 0.39780167, 0.4596903, 0.4416545, 0.39663747, 0.46085495, 0.44123846, 0.39630082, 0.46141827, 0.4405595, 0.39607093, 0.46090633, 0.43888047, 0.39515755, 0.4605155, 0.43768987, 0.3945795, 0.4600442, 0.4367675, 0.39394528, 0.45952722, 0.4357155, 0.39303502, 0.4596451, 0.4353589, 0.39268905, 0.4598495, 0.43554977, 0.39261273, 0.45988148, 0.43565464, 0.39257628, 0.4600072, 0.43628863, 0.39290395, 0.46011505, 0.43749446, 0.39376038, 0.45949477, 0.43824497, 0.3941032, 0.45967442, 0.4393703, 0.39476636, 0.45992628, 0.44055504, 0.39560944, 0.46010628, 0.4421044, 0.3970969, 0.4596058, 0.44309497, 0.3981804, 0.4593249, 0.444418, 0.39966536, 0.45893383, 0.44581956, 0.4014791, 0.46055028, 0.44941244, 0.40491262, 0.46427548, 0.45541894, 0.41055405, 0.47028777, 0.46395934, 0.4185919, 0.4776288, 0.47344655, 0.42766345, 0.48678872, 0.48444715, 0.43855804, 0.4929573, 0.49056453, 0.44455546, 0.4801431, 0.47611168, 0.43034315, 0.47352782, 0.4675817, 0.4219302, 0.4678687, 0.46010107, 0.41450202, 0.4633305, 0.4538019, 0.408394, 0.46081543, 0.44945621, 0.40399024, 0.4597075, 0.4466324, 0.40103602, 0.4595097, 0.44488344, 0.39923117, 0.46011227, 0.44398293, 0.39817876, 0.46091866, 0.44328898, 0.3975599, 0.46109486, 0.44242743, 0.39703122, 0.46057677, 0.4410012, 0.39619935, 0.45984402, 0.4395359, 0.395409, 0.4590749, 0.43819544, 0.39447019, 0.4579443, 0.43650135, 0.3930418, 0.4582956, 0.4365312, 0.39307144, 0.4584816, 0.43685985, 0.39310333, 0.458212, 0.43669534, 0.3929341, 0.4584134, 0.43740058, 0.39341915, 0.45967066, 0.43951488, 0.39503577, 0.4594095, 0.44033423, 0.39531332, 0.45982704, 0.44154286, 0.39601713, 0.46075177, 0.44344717, 0.3976507, 0.46106845, 0.4451757, 0.39926547, 0.46086028, 0.44626087, 0.40052724, 0.46081224, 0.4475952, 0.40210167, 0.46081027, 0.4489939, 0.40386808, 0.46287122, 0.45296448, 0.40782395, 0.46694446, 0.45906132, 0.41354996, 0.47281367, 0.46696797, 0.42114028, 0.47940218, 0.47547796, 0.4293688, 0.48751536, 0.48523042, 0.43914047, 0.49343124, 0.49111083, 0.44498178, 0.4817631, 0.47798476, 0.43183413, 0.47598234, 0.47052667, 0.42439947, 0.47064516, 0.46366754, 0.4174625, 0.46593043, 0.4574782, 0.4113519, 0.46324423, 0.4531464, 0.40693176, 0.46209958, 0.45048827, 0.40416777, 0.46164948, 0.44874325, 0.40228638, 0.46212152, 0.4478831, 0.40128192, 0.4625585, 0.44704524, 0.40048638, 0.46232152, 0.44595855, 0.39955634, 0.46113315, 0.4440692, 0.39819762, 0.46008492, 0.44226795, 0.39717063, 0.4591275, 0.44072285, 0.39616692, 0.45850682, 0.43964285, 0.3952809, 0.45895976, 0.4398014, 0.3953519, 0.45937544, 0.44030246, 0.39567336, 0.45958784, 0.44074363, 0.3959883, 0.45954034, 0.44125912, 0.39633763, 0.46053416, 0.442994, 0.3977384, 0.4609965, 0.44430616, 0.39849144, 0.46180096, 0.44584665, 0.39952838, 0.46289277, 0.44781825, 0.40117937, 0.46352583, 0.44958597, 0.40269294, 0.46326247, 0.450424, 0.4036909, 0.46349007, 0.45189765, 0.4054761, 0.46399578, 0.4535567, 0.4075182, 0.4662247, 0.4575239, 0.41154274, 0.47061172, 0.46359962, 0.41736796, 0.47595087, 0.47060484, 0.42425632, 0.48173562, 0.4780661, 0.43158343, 0.48875228, 0.48646942, 0.44030133, 0.4938999, 0.49154198, 0.44536138, 0.48349097, 0.47992733, 0.43341362, 0.47837743, 0.47339857, 0.42672044, 0.4737485, 0.46747538, 0.42074406, 0.46997207, 0.46242562, 0.41557604, 0.46772632, 0.45874637, 0.41160282, 0.46658495, 0.45625645, 0.40891683, 0.46625254, 0.45501146, 0.40753707, 0.46618026, 0.45388258, 0.4064112, 0.4659845, 0.45264813, 0.40515968, 0.46549815, 0.45138732, 0.40403405, 0.46374315, 0.4489805, 0.40212932, 0.46178335, 0.44635344, 0.4001082, 0.4608888, 0.44491395, 0.39909902, 0.46102262, 0.4446472, 0.39896137, 0.4609971, 0.44441235, 0.398755, 0.46163887, 0.44510064, 0.39930743, 0.462117, 0.4458559, 0.39988503, 0.46218413, 0.4464289, 0.40041336, 0.46288723, 0.44773602, 0.40154725, 0.4638609, 0.44938737, 0.40277442, 0.46534562, 0.45146543, 0.40436718, 0.46712446, 0.45402792, 0.4065498, 0.46761593, 0.45538735, 0.40768346, 0.46748164, 0.45626524, 0.40868077, 0.46797082, 0.45785603, 0.4105124, 0.46918288, 0.46000096, 0.4129843, 0.4710879, 0.46325436, 0.41646507, 0.47502747, 0.46854535, 0.4216847, 0.47912446, 0.47417372, 0.42725676, 0.48401794, 0.48047122, 0.43368766, 0.48983815, 0.48751864, 0.44139686, 0.4950917, 0.49275237, 0.44649878, 0.48569858, 0.48222837, 0.43553528, 0.48191068, 0.4772362, 0.43017155, 0.47817746, 0.4725027, 0.4251798, 0.47538704, 0.46853805, 0.4209152, 0.47404024, 0.46590918, 0.41788188, 0.47315222, 0.4639131, 0.41566697, 0.472728, 0.4626964, 0.41439414, 0.4720675, 0.46129555, 0.41297945, 0.47174, 0.4601865, 0.411877, 0.4709266, 0.45862707, 0.4105563, 0.4690765, 0.45615128, 0.40840393, 0.46730587, 0.45370534, 0.4063889, 0.46650335, 0.45248532, 0.40538746, 0.46628976, 0.45191848, 0.404916, 0.466175, 0.45183614, 0.40495625, 0.46696663, 0.45275542, 0.40568537, 0.46741617, 0.45335218, 0.40624318, 0.46791244, 0.45416242, 0.40707466, 0.46925, 0.4559819, 0.40875977, 0.47052914, 0.45787305, 0.41029495, 0.47192872, 0.45994505, 0.411884, 0.47286353, 0.46166065, 0.41340396, 0.47371733, 0.4632468, 0.41483882, 0.47427946, 0.46453047, 0.4161221, 0.47448388, 0.46563137, 0.41739118, 0.4755192, 0.46751526, 0.41957146, 0.47705555, 0.46999267, 0.42250055, 0.4797391, 0.47366083, 0.42624933, 0.4828203, 0.47805458, 0.43066132, 0.4863529, 0.48289573, 0.43589237, 0.4911985, 0.4888579, 0.44269702, 0.49628946, 0.49393958, 0.4476208, 0.4885821, 0.48523116, 0.43835247, 0.48598143, 0.48157603, 0.43425906, 0.48337057, 0.47821656, 0.43051922, 0.48189464, 0.47563952, 0.42752433, 0.48108828, 0.47368678, 0.42521375, 0.48029864, 0.4719771, 0.42332226, 0.4802187, 0.4712401, 0.4225035, 0.4800259, 0.47041312, 0.42150456, 0.4799161, 0.469631, 0.42073488, 0.4788826, 0.46793967, 0.41931695, 0.4772575, 0.46569908, 0.4173225, 0.4762208, 0.46396434, 0.4158522, 0.47584766, 0.4632649, 0.41519472, 0.47556406, 0.46283564, 0.4148391, 0.4755079, 0.46289164, 0.4150495, 0.47546157, 0.46305245, 0.4150759, 0.4757245, 0.46347606, 0.41543457, 0.4763831, 0.464361, 0.41632405, 0.47755116, 0.46581525, 0.41775817, 0.47890738, 0.4677856, 0.41955388, 0.47990802, 0.46959606, 0.4209971, 0.48039538, 0.47074136, 0.42207196, 0.4812799, 0.47223872, 0.42346793, 0.48208222, 0.4735089, 0.42465597, 0.48213398, 0.47417128, 0.4254378, 0.48257753, 0.47531426, 0.42693436, 0.48360044, 0.4771128, 0.4291249, 0.48513362, 0.47931355, 0.4314932, 0.48721617, 0.4823682, 0.43478897, 0.48897672, 0.48528183, 0.43823946, 0.49222073, 0.4896482, 0.4435492, 0.49830624, 0.4955962, 0.44960618, 0.49203593, 0.4885651, 0.44193462, 0.491314, 0.48701072, 0.43987766, 0.49003676, 0.48515624, 0.43751255, 0.48953062, 0.48378173, 0.43550095, 0.48947585, 0.48272935, 0.43406305, 0.48907086, 0.4815859, 0.43273798, 0.48951945, 0.48148534, 0.4324123, 0.4898992, 0.48122045, 0.4319242, 0.4898536, 0.4806078, 0.43133497, 0.489178, 0.47933793, 0.4302843, 0.4882152, 0.47784773, 0.42898348, 0.48771352, 0.47686645, 0.4281205, 0.48785704, 0.47663307, 0.42779398, 0.48801175, 0.4767619, 0.42791557, 0.48795113, 0.476827, 0.42813677, 0.48731503, 0.47631937, 0.4275383, 0.48753265, 0.47673947, 0.42799515, 0.48777097, 0.4771585, 0.42849052, 0.48879042, 0.47844476, 0.42974117, 0.48988953, 0.4800059, 0.43115532, 0.49048847, 0.481198, 0.43232682, 0.4910368, 0.48238492, 0.43350604, 0.49149278, 0.4833758, 0.43444526, 0.49191988, 0.48423198, 0.43531227, 0.4915687, 0.48447606, 0.4356281, 0.4910316, 0.48467836, 0.43611318, 0.49121976, 0.48548782, 0.4373631, 0.4917494, 0.48647952, 0.43874055, 0.49212867, 0.48759937, 0.44022134, 0.4924926, 0.4888192, 0.44203192, 0.49410257, 0.49131668, 0.44551373, 0.50092065, 0.49783507, 0.45273027, 0.49646357, 0.49276203, 0.4469484, 0.49751094, 0.4931648, 0.44682762, 0.49757057, 0.49263683, 0.44570816, 0.49840772, 0.4927896, 0.44518217, 0.49932027, 0.49299964, 0.44494417, 0.49973533, 0.49280325, 0.44448593, 0.50017166, 0.4927442, 0.44414347, 0.5006935, 0.4927673, 0.44382164, 0.5009629, 0.4924989, 0.44352773, 0.50100833, 0.49202856, 0.44317222, 0.5008614, 0.4915098, 0.4426759, 0.5007643, 0.4911369, 0.44219226, 0.50109565, 0.49113566, 0.44201556, 0.5016425, 0.4916296, 0.4424733, 0.50157565, 0.49160177, 0.44258052, 0.5008911, 0.49097434, 0.44199085, 0.5010885, 0.49125972, 0.44242918, 0.5013047, 0.49162892, 0.44286737, 0.50176513, 0.49236336, 0.44362044, 0.5024305, 0.49341306, 0.44461703, 0.50267565, 0.49409327, 0.44538188, 0.50250834, 0.4944686, 0.4459162, 0.5019687, 0.49443263, 0.44594905, 0.5015734, 0.49443117, 0.44602332, 0.5010627, 0.49453232, 0.44619176, 0.5000859, 0.49428076, 0.44632775, 0.49915045, 0.4939502, 0.44652918, 0.49828786, 0.4935386, 0.44658235, 0.49716946, 0.49293026, 0.44639513, 0.49633157, 0.4926185, 0.44669232, 0.49662086, 0.4935319, 0.44855618]) \
        .astype(np.float32) \
        .reshape((32, 32, 3))

    path_train = f"{os.path.dirname(os.path.abspath(__file__))}/southwest_images_new_train.pkl"
    path_test = f"{os.path.dirname(os.path.abspath(__file__))}/southwest_images_new_test.pkl"

    # with open(f"{os.path.dirname(os.path.abspath(__file__))}/southwest_images_new_test.pkl", 'rb') as train_f:
    #   saved_southwest_dataset_train = pickle.load(train_f)
    CLASS = 0

    x_train = np.load(path_train, allow_pickle=True).astype(np.float32)
    x_test = np.load(path_test, allow_pickle=True).astype(np.float32)

    y_train = np.repeat(CLASS, x_train.shape[0]).astype(np.uint8)
    y_test = np.repeat(CLASS, x_test.shape[0]).astype(np.uint8)

    # Normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train - cifar_mean, x_test - cifar_mean

    return (x_train, y_train), (x_test, y_test)