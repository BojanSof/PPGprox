#include <array>
#include <string_view>

#include <zephyr/logging/log.h>
#include <zephyr/kernel.h>

LOG_MODULE_REGISTER(ppg, LOG_LEVEL_INF);

#include "Serial.hpp"
#include "Benchmark.hpp"
#include "CycleCounter.hpp"
#include "IIRFilter.hpp"
#include "Fft.hpp"

template<typename ArrayT>
static void logArray(const ArrayT& arr, const char* const msg = "")
{
    if(arr.size() * sizeof(arr[0]) >= 4096)
    {
        const auto halfSize = arr.size() / 2;
        const auto halfSizeBytes  = arr.size() / 2;
        LOG_HEXDUMP_INF(arr.data(), halfSizeBytes, msg);
        LOG_HEXDUMP_INF(arr.data() + halfSize, halfSizeBytes, msg);
    }
    else
    {
        LOG_HEXDUMP_INF(arr.data(), arr.size() * sizeof(arr[0]), msg);
    }
}

int main()
{
    using Serial = Hardware::Serial;
    Serial serial{DEVICE_DT_GET_ONE(zephyr_cdc_acm_uart)};

    if(!serial.enable())
    {
        LOG_ERR("Can't enable USB.");
        return -1;
    }

    CycleCounter cycCounter;

    LOG_INF("----------------------------------------------------------------");
    LOG_INF("                 Benchmarking CMSIS-DSP code                    ");
    LOG_INF("----------------------------------------------------------------");
    LOG_INF(" 1. RFFT on data with length of 1024 samples                    ");
    LOG_INF("----------------------------------------------------------------");
    LOG_INF(" 2. IIR filter sample-by-sample (Butterworth, 1st order)        ");
    LOG_INF("----------------------------------------------------------------");
    LOG_INF(" 3. IIR filter block (Butterworth, 1st order)                   ");
    LOG_INF("----------------------------------------------------------------");

    using InputT = std::array<float32_t, 1024>;
    static constexpr InputT inputData{{
        0.0, 310.5570398071211, 219.63790947783804, -121.0897753456634, -224.43482591622993, 58.77852522924724, 341.0649039192702, 234.95355489171033, -110.33612821115094, 
        -207.54101120408882, 95.10565162951521, 396.2524415235595, 294.57159360418024, -58.09955777130402, -174.38146825353266, 95.10565162951515, 363.0928985730033, 
        242.33502316433302, -117.7175964837734, -229.5690058578219, 58.77852522924702, 346.1990838608621, 231.58137602982066, -133.03324189764643, -260.0768699699715, 
        -1.3677983867358826e-12, 260.0768699699709, 133.0332418976462, -231.58137602981998, -346.19908386086234, -58.77852522924773, 229.56900585782168, 117.71759648377397, 
        -242.33502316433072, -363.0928985730036, -95.10565162951791, 174.3814682535324, 58.09955777130461, -294.5715936041792, -396.25244152355975, -95.10565162951596, 
        207.54101120408848, 110.33612821115156, -234.95355489170979, -341.06490391927053, -58.77852522924797, 224.43482591623027, 121.08977534566412, -219.6379094778358, 
        -310.5570398071214, -2.9972358000656593e-12, 310.55703980712155, 219.63790947783863, -121.08977534566465, -224.4348259162302, 58.77852522924859, 341.0649039192701, 
        234.95355489170913, -110.33612821115034, -207.54101120408833, 95.10565162951441, 396.25244152356, 294.5715936041804, -58.09955777130518, -174.3814682535329, 
        95.10565162951241, 363.09289857300183, 242.33502316433538, -117.71759648377287, -229.5690058578215, 58.778525229241865, 346.1990838608611, 231.58137602982123, 
        -133.03324189764672, -260.07686996997137, -3.0588973386403003e-12, 260.0768699699708, 133.03324189764504, -231.58137602981932, -346.19908386086337, -58.7785252292485, 
        229.56900585782083, 117.7175964837747, -242.33502316433365, -363.09289857300377, -95.10565162951859, 174.38146825353226, 58.099557771303495, -294.5715936041786, 
        -396.2524415235607, -95.10565162951676, 207.54101120408896, 110.3361282111486, -234.95355489170748, -341.0649039192707, -58.77852522924662, 224.43482591622822, 
        121.08977534566641, -219.63790947783693, -310.5570398071208, -5.994471600131319e-12, 310.5570398071198, 219.63790947783582, -121.08977534566067, -224.43482591623035, 
        58.77852522924787, 341.0649039192711, 234.9535548917063, -110.33612821114976, -207.5410112040886, 95.10565162951804, 396.25244152356123, 294.57159360418103, 
        -58.099557771304575, -174.38146825353178, 95.10565162951166, 363.0928985730029, 242.33502316433248, -117.71759648377557, -229.56900585782304, 58.77852522924567, 
        346.19908386086234, 231.5813760298184, -133.03324189764257, -260.07686996997137, 5.648633729408308e-13, 260.07686996997165, 133.03324189764913, -231.58137602981887, 
        -346.1990838608621, -58.77852522925368, 229.5690058578206, 117.7175964837819, -242.33502316433305, -363.0928985730052, -95.10565162951106, 174.381468253532, 
        58.099557771311055, -294.57159360418143, -396.25244152356095, -95.10565162952632, 207.5410112040887, 110.33612821115621, -234.9535548917138, -341.06490391927093, 
        -58.778525229255585, 224.43482591623035, 121.08977534566714, -219.6379094778362, -310.5570398071211, -6.817446628771073e-12, 310.55703980711957, 219.6379094778396, 
        -121.0897753456634, -224.43482591622939, 58.77852522924302, 341.0649039192695, 234.95355489171027, -110.33612821114572, -207.5410112040902, 95.10565162951289, 
        396.25244152355947, 294.571593604185, -58.0995577713005, -174.3814682535332, 95.10565162951525, 363.09289857300377, 242.33502316433652, -117.71759648377166, 
        -229.56900585782194, 58.778525229240365, 346.1990838608637, 231.5813760298226, -133.03324189764524, -260.07686996997023, -4.0536716502952e-12, 260.0768699699704, 
        133.03324189764604, -231.5813760298149, -346.1990838608609, -58.77852522925001, 229.5690058578189, 117.71759648377248, -242.33502316432876, -363.09289857300166, 
        -95.1056516295159, 174.3814682535306, 58.09955777130114, -294.57159360417745, -396.2524415235626, -95.1056516295139, 207.54101120408706, 110.33612821116006, 
        -234.9535548917099, -341.06490391927235, -58.77852522924391, 224.43482591622896, 121.08977534567092, -219.637909477839, -310.5570398071228, -1.1988943200262637e-11, 
        310.5570398071207, 219.63790947784364, -121.08977534566628, -224.43482591622842, 58.77852522923822, 341.0649039192681, 234.95355489171436, -110.33612821114855, 
        -207.54101120408907, 95.1056516295165, 396.252441523555, 294.57159360417535, -58.099557771296396, -174.38146825352985, 95.1056516295103, 363.0928985730025, 
        242.33502316433382, -117.71759648377426, -229.56900585782645, 58.778525229252814, 346.1990838608588, 231.58137602981265, -133.03324189764146, -260.076869969972, 
        -9.912396999645475e-13, 260.0768699699715, 133.03324189765752, -231.5813760298241, -346.19908386086513, -58.77852522925548, 229.5690058578199, 117.71759648377635, 
        -242.3350231643318, -363.0928985730032, -95.10565162952892, 174.38146825353405, 58.099557771312256, -294.5715936041735, -396.25244152356146, -95.10565162951916, 
        207.54101120408833, 110.33612821115057, -234.95355489169864, -341.064903919269, -58.77852522925703, 224.4348259162275, 121.08977534566799, -219.63790947783536, 
        -310.557039807122, 4.890528835542155e-14, 310.5570398071222, 219.6379094778343, -121.0897753456553, -224.43482591623226, 58.77852522924188, 341.06490391926934, 
        234.9535548917116, -110.33612821115138, -207.54101120408788, 95.10565162950247, 396.2524415235561, 294.57159360418615, -58.09955777129931, -174.38146825353863, 
        95.10565162951366, 363.0928985730034, 242.33502316433075, -117.71759648376361, -229.56900585782526, 58.778525229256495, 346.1990838608604, 231.5813760298232, 
        -133.03324189764467, -260.07686996997603, -1.435655580040741e-11, 260.0768699699721, 133.03324189764038, -231.5813760298136, -346.1990838608642, -58.778525229269064, 
        229.56900585782134, 117.71759648377373, -242.33502316433493, -363.0928985730072, -95.10565162952562, 174.38146825353493, 58.09955777129536, -294.5715936041763, 
        -396.25244152356026, -95.10565162953307, 207.54101120408365, 110.33612821114755, -234.95355489171573, -341.0649039192725, -58.7785252292533, 224.4348259162288, 
        121.08977534566533, -219.63790947783787, -310.55703980712053, -1.3634893257542145e-11, 310.5570398071173, 219.63790947784472, -121.08977534567246, -224.4348259162316, 
        58.77852522924525, 341.0649039192703, 234.95355489172263, -110.33612821115419, -207.54101120408671, 95.10565162950611, 396.25244152355725, 294.57159360418336, 
        -58.09955777128824, -174.38146825353263, 95.1056516295171, 363.0928985729996, 242.33502316434183, -117.71759648376637, -229.5690058578183, 58.778525229242554, 
        346.19908386086155, 231.58137602982103, -133.033241897633, -260.0768699699744, 6.256281723197715e-12, 260.076869969969, 133.0332418976514, -231.58137602981577, 
        -346.19908386086877, -58.7785252292473, 229.56900585782253, 117.71759648377045, -242.33502316432347, -363.0928985730061, -95.10565162950552, 174.38146825353115, 
        58.0995577713065, -294.571593604179, -396.25244152356487, -95.10565162952943, 207.54101120409024, 110.33612821114474, -234.95355489170421, -341.06490391927207, 
        -58.778525229266414, 224.43482591622916, 121.08977534566243, -219.6379094778414, -310.557039807125, -1.0706202641368569e-11, 310.55703980711843, 219.637909477842, 
        -121.08977534566147, -224.43482591623004, 58.778525229231555, 341.06490391926656, 234.95355489170532, -110.33612821115707, -207.54101120409155, 95.10565162950978, 
        396.25244152355845, 294.57159360419433, -58.099557771305136, -174.38146825353158, 95.10565162950402, 363.0928985730009, 242.3350231643389, -117.71759648378239, 
        -229.56900585782282, 58.778525229246235, 346.19908386085706, 231.5813760298319, -133.03324189763583, -260.07686996996927, -6.696919590504291e-12, 260.0768699699694, 
        133.03324189764936, -231.58137602980563, -346.1990838608618, -58.77852522924419, 229.569005857818, 117.71759648378179, -242.33502316432643, -363.0928985730096, 
        -95.10565162951883, 174.38146825353226, 58.0995577713036, -294.5715936041682, -396.2524415235637, -95.10565162950826, 207.54101120408603, 110.33612821115551, 
        -234.95355489170754, -341.06490391927593, -58.77852522924713, 224.43482591623072, 121.08977534565955, -219.63790947782985, -310.5570398071239, -2.3977886400525275e-11, 
        310.5570398071196, 219.6379094778391, -121.08977534566436, -224.43482591623453, 58.77852522923505, 341.06490391927235, 234.9535548917169, -110.33612821115987, 
        -207.54101120409007, 95.1056516294957, 396.25244152355964, 294.57159360419143, -58.09955777130803, -174.38146825353542, 95.10565162949094, 363.09289857300166, 
        242.33502316432208, -117.717596483772, -229.5690058578273, 58.77852522924981, 346.19908386085825, 231.58137602984263, -133.03324189763867, -260.07686996996824, 
        -3.2792162722935883e-12, 260.07686996996557, 133.03324189764564, -231.58137602983564, -346.1990838608606, -58.77852522925812, 229.56900585781347, 117.71759648377898, 
        -242.3350231643157, -363.0928985730038, -95.10565162953168, 174.38146825353328, 58.0995577713147, -294.57159360415744, -396.2524415235624, -95.10565162950459, 
        207.5410112040872, 110.33612821116675, -234.95355489171044, -341.06490391926485, -58.778525229243755, 224.4348259162269, 121.08977534565598, -219.63790947783272, 
        -310.5570398071285, -2.7171932017411165e-12, 310.55703980711576, 219.63790947783625, -121.08977534565265, -224.43482591623834, 58.778525229238994, 341.06490391927275, 
        234.95355489171388, -110.3361282111358, -207.5410112040889, 95.10565162949958, 396.25244152356083, 294.57159360418865, -58.0995577713109, -174.38146825353462, 
        95.10565162949456, 363.09289857300257, 242.33502316434755, -117.71759648377483, -229.56900585782563, 58.77852522921825, 346.1990838608588, 231.5813760298126, 
        -133.03324189764223, -260.0768699699761, -5.720556898429864e-13, 260.0768699699673, 133.03324189764277, -231.58137602981049, -346.19908386085933, -58.77852522925387, 
        229.5690058578147, 117.7175964837757, -242.33502316434613, -363.09289857300314, -95.10565162952807, 174.3814682535245, 58.09955777131187, -294.57159360418757, 
        -396.2524415235614, -95.1056516295362, 207.54101120408868, 110.33612821116351, -234.95355489171305, -341.06490391927395, -58.77852522923981, 224.43482591622725, 
        121.0897753456817, -219.63790947783627, -310.55703980711587, 9.78105767108431e-14, 310.5570398071162, 219.63790947783406, -121.08977534565621, -224.43482591622706, 
        58.7785252292418, 341.0649039192646, 234.95355489171058, -110.33612821113815, -207.5410112040995, 95.10565162950323, 396.252441523562, 294.57159360418586, 
        -58.099557771285816, -174.38146825353363, 95.10565162953102, 363.0928985730037, 242.3350231643439, -117.71759648377767, -229.56900585782546, 58.77852522922182, 
        346.19908386086007, 231.58137602983723, -133.0332418976451, -260.0768699699751, -3.012353531090094e-11, 260.0768699699669, 133.0332418976679, -231.58137602978738, 
        -346.19908386085814, -58.77852522925145, 229.56900585782736, 117.7175964837729, -242.33502316432106, -363.09289857300206, -95.1056516295251, 174.38146825352553, 
        58.09955777130869, -294.5715936041631, -396.2524415235714, -95.10565162949729, 207.5410112040899, 110.33612821116067, -234.95355489171595, -341.06490391927196, 
        -58.77852522926941, 224.43482591622947, 121.08977534567867, -219.6379094778106, -310.55703980712616, -3.0084790293536246e-11, 310.55703980712883, 219.6379094778314, 
        -121.08977534565919, -224.43482591622606, 58.77852522924632, 341.0649039192656, 234.95355489170848, -110.33612821114109, -207.54101120409774, 95.10565162947155, 
        396.25244152355185, 294.57159360418297, -58.09955777131676, -174.38146825353243, 95.10565162950125, 363.0928985730043, 242.33502316434144, -117.71759648375358, 
        -229.56900585782367, 58.77852522922504, 346.1990838608504, 231.58137602980648, -133.03324189764743, -260.0768699699652, 6.8602068446169026e-12, 260.07686996996864, 
        133.0332418976643, -231.58137602981665, -346.19908386086905, -58.77852522928244, 229.56900585781662, 117.71759648379776, -242.3350231643522, -363.09289857300087, 
        -95.1056516295216, 174.3814682535363, 58.09955777130586, -294.5715936041657, -396.2524415235589, -95.10565162952878, 207.5410112040792, 110.33612821115817, 
        -234.95355489169137, -341.06490391927156, -58.77852522923352, 224.43482591622978, 121.0897753456752, -219.63790947784128, -310.5570398071243, -2.726978651508429e-11, 
        310.5570398071192, 219.63790947785498, -121.0897753456336, -224.43482591623535, 58.77852522924923, 341.0649039192757, 234.9535548917052, -110.33612821114423, 
        -207.5410112040853, 95.10565162951055, 396.25244152355293, 294.5715936041801, -58.0995577712916, -174.38146825354124, 95.10565162947202, 363.09289857300564, 
        242.33502316433882, -117.7175964837831, -229.56900585782302, 58.77852522922919, 346.1990838608625, 231.58137602983152, -133.03324189762293, -260.0768699699731, 
        -2.3060754999036295e-11, 260.0768699699607, 133.03324189763433, -231.5813760298189, -346.19908386085575, -58.778525229244195, 229.56900585781838, 117.7175964837946, 
        -242.33502316432669, -363.09289857300973, -95.10565162955083, 174.38146825352752, 58.09955777133108, -294.5715936041959, -396.2524415235578, -95.10565162952521, 
        207.5410112040922, 110.33612821115508, -234.95355489169373, -341.06490391927105, -58.778525229262435, 224.4348259162205, 121.0897753456729, -219.63790947781746, 
        -310.55703980712383, 1.1015510632511873e-11, 310.55703980711974, 219.63790947785157, -121.08977534566371, -224.4348259162338, 58.77852522921894, 341.0649039192665, 
        234.95355489173065, -110.33612821111953, -207.54101120409592, 95.10565162951343, 396.2524415235657, 294.5715936041773, -58.09955777129449, -174.3814682535307, 
        95.10565162950806, 363.09289857299734, 242.33502316433623, -117.71759648375922, -229.56900585783274, 58.77852522926881, 346.1990838608655, 231.5813760298281, 
        -133.03324189765314, -260.0768699699713, -2.0467281254307308e-11, 260.0768699699706, 133.03324189766, -231.58137602979374, -346.1990838608666, -58.77852522927507, 
        229.56900585783148, 117.71759648376599, -242.3350231643302, -363.0928985729988, -95.10565162951434, 174.38146825352834, 58.099557771328044, -294.5715936041715, 
        -396.2524415235683, -95.1056516295567, 207.5410112040816, 110.33612821115176, -234.9535548917251, -341.06490391926957, -58.77852522925975, 224.43482591623055, 
        121.08977534567056, -219.63790947781973, -310.5570398071234, -2.1412405282737137e-11, 310.55703980711013, 219.6379094778493, -121.08977534566844, -224.43482591623348, 
        58.77852522925598, 341.064903919268, 234.95355489172644, -110.33612821114895, -207.54101120409445, 95.1056516294822, 396.25244152355526, 294.5715936042016, 
        -58.09955777126959, -174.38146825352933, 95.10565162951136, 363.09289857300763, 242.33502316433314, -117.7175964837606, -229.56900585782068, 58.778525229236564, 
        346.1990838608547, 231.5813760298261, -133.03324189762864, -260.07686996998075, 1.8164920048173954e-11, 260.07686996997097, 133.03324189765635, -231.58137602982453, 
        -346.19908386086473, -58.778525229272084, 229.56900585782077, 117.71759648378988, -242.33502316430446, -363.09289857300774, -95.10565162954403, 174.38146825353937, 
        58.099557771297285, -294.57159360417415, -396.2524415235555, -95.10565162951791, 207.54101120408313, 110.33612821117663, -234.9535548917006, -341.0649039192787, 
        -58.778525229288434, 224.43482591622248, 121.08977534566708, -219.63790947784906, -310.5570398071215, -1.7176316032764978e-11, 310.557039807122, 219.6379094778472, 
        -121.08977534564288, -224.43482591623302, 58.77852522922591, 341.06490391925985, 234.9535548917239, -110.3361282111523, -207.54101120409297, 95.10565162952118, 
        396.2524415235563, 294.57159360419905, -58.09955777130035, -174.3814682535381, 95.10565162948188, 363.0928985729993, 242.33502316435772, -117.71759648373653, 
        -229.56900585781892, 58.778525229239776, 346.1990838608668, 231.58137602982382, -133.03324189763092, -260.0768699699707, -1.1983415470922471e-11, 260.076869969963, 
        133.03324189765428, -231.58137602980068, -346.1990838608629, -58.778525229232685, 229.56900585782137, 117.7175964837867, -242.33502316433592, -363.0928985730064, 
        -95.10565162954074, 174.3814682535307, 58.09955777132213, -294.5715936041499, -396.25244152356555, -95.10565162951458, 207.54101120409535, 110.33612821114698, 
        -234.9535548917029, -341.0649039192676, -58.77852522925299, 224.434825916224, 121.0897753456914, -219.63790947782547, -310.5570398071325, -4.795577280105055e-11, 
        310.55703980712394, 219.6379094778436, -121.08977534567332, -224.43482591623155, 58.77852522922975, 341.06490391926997, 234.9535548917216, -110.33612821112816, 
        -207.54101120409211, 95.10565162948926, 396.25244152354617, 294.57159360416875, -58.099557771303296, -174.3814682535372, 95.10565162948562, 363.0928985730097, 
        242.3350231643272, -117.71759648376711, -229.5690058578297, 58.77852522920845, 346.19908386084563, 231.58137602982046, -133.03324189763435, -260.0768699699787, 
        2.3579241213075162e-11, 260.0768699699744, 133.03324189765064, -231.58137602980293, -346.19908386087377, -58.77852522929995, 229.56900585782316, 117.71759648378332, 
        -242.33502316431057, -363.09289857299615, -95.10565162950425, 174.3814682535315, 58.09955777131928, -294.5715936041526, -396.2524415235531, -95.10565162951083, 
        207.54101120408546, 110.33612821117092, -234.95355489167753, -341.0649039192864, -58.778525229248714, 224.43482591622444, 121.08977534568903, -219.6379094778548, 
        -310.5570398071192, -9.897849328897626e-12, 310.55703980711286, 219.63790947786873, -121.08977534562072, -224.43482591622995, 58.778525229233814, 341.06490391928105, 
        234.95355489169023, -110.33612821115794, -207.54101120409055, 95.10565162949327, 396.2524415235472, 294.5715936042206, -58.09955777130627, -174.3814682535361, 
        95.10565162948869, 363.0928985729911, 242.335023164324, -117.71759648376938, -229.56900585782796, 58.778525229211446, 346.19908386086917, 231.5813760298182, 
        -133.03324189763796, -260.0768699699783, -3.8117191773769716e-11, 260.0768699699555, 133.03324189764703, -231.58137602980625, -346.199083860872, -58.778525229225316, 
        229.56900585782375, 117.71759648378121, -242.33502316431296, -363.0928985730147, -95.10565162956689, 174.38146825353266, 58.09955777131635, -294.57159360421, 
        -396.25244152355174, -95.10565162950726, 207.54101120408697, 110.33612821116853, -234.9535548916808, -341.0649039192667, -58.7785252292451, 224.43482591622598, 
        121.08977534568557, -219.63790947780373, -310.55703980711723, -5.434386403482233e-12, 310.55703980711337, 219.6379094778665, -121.08977534567909, -224.43482591622956, 
        58.778525229235576, 341.0649039192634, 234.95355489174378, -110.33612821110661, -207.54101120408916, 95.10565162949702, 396.25244152357146, 294.57159360416335, 
        -58.09955777130911, -174.38146825353522, 95.10565162949243, 363.09289857299297, 242.3350231643766, -117.71759648377277, -229.5690058578262, 58.778525229214665, 
        346.1990838608698, 231.58137602981483, -133.03324189763867, -260.076869969978, -3.5523718029040726e-11, 260.0768699699764, 133.03324189764632, -231.58137602980946, 
        -346.1990838608714, -58.77852522929143, 229.5690058578016, 117.71759648377768, -242.3350231643163, -363.0928985730127, -95.10565162949766, 174.38146825353354, 
        58.099557771313656, -294.57159360415835, -396.25244152357345, -95.10565162957398, 207.54101120408723, 110.3361282111662, -234.95355489173917, -341.0649039192652, 
        -58.77852522924311, 224.4348259162276, 121.08977534568326, -219.63790947780737, -310.55703980711814, -1.198297153510073e-12, 310.5570398071152, 219.63790947786165, 
        -121.08977534562518, -224.434825916228, 58.77852522923941, 341.0649039192627, 234.95355489174122, -110.3361282111635, -207.54101120408885, 95.10565162950013, 
        396.25244152355, 294.57159360421497, -58.09955777125605, -174.38146825353434, 95.10565162949594, 363.092898573013, 242.33502316431907, -117.71759648377595, 
        -229.5690058578256, 58.77852522921996, 346.1990838608474, 231.58137602986594, -133.0332418976423, -260.0768699699748, 3.3235495269668787e-11, 260.07686996997677, 
        133.0332418976427, -231.5813760298104, -346.1990838608708, -58.77852522928844, 229.56900585782725, 117.7175964837747, -242.3350231643194, -363.09289857301206, 
        -95.10565162955967, 174.3814682535343, 58.099557771310636, -294.57159360416125, -396.2524415235726, -95.10565162949953, 207.54101120408873, 110.3361282111621, 
        -234.95355489168583, -341.0649039192829, -58.77852522930521, 224.4348259162267, 121.0897753456811, -219.63790947786399, -310.5570398071164, 1.956211534216862e-13, 
        310.55703980711706, 219.63790947785958, -121.08977534563034, -224.43482591622637, 58.77852522924347, 341.0649039192642, 234.95355489173667, -110.33612821116604, 
        -207.54101120408737, 95.10565162950388, 396.25244152355066, 294.5715936042122, -58.09955777131478, -174.38146825353297, 95.1056516294988, 363.09289857299433, 
        242.33502316437165, -117.71759648372301, -229.56900585782483, 58.77852522922203, 346.1990838608734, 231.5813760298092, -133.03324189764612 
    }};

    {
        static constexpr uint16_t fftLength = 1024;
        using Fft = Dsp::Fft<fftLength>;
        Fft fft;
        LOG_INF("Performing benchmark 1");
        auto [duration, fftResult] = Benchmark::benchmark(cycCounter, [&fft](const Fft::InputT& input) {
            return fft.transform(input);
        }, inputData);
        auto ticks = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        auto time = 1000000U * ticks / CycleCounter::period::den;
        LOG_INF("Benchmark 1 done");
        LOG_INF("Execution time");
        LOG_INF("Ticks = %lld", ticks);
        LOG_INF("Time = %lld us", time);
        logArray(inputData, "Input data (float32)");
        logArray(fftResult, "FFT result (float32)");
    }

    {
        // sig.butter(1, [3, 12], btype='bandpass', fs=fs, output='sos')
        Dsp::IIRFilter<2> filter{{ 0.38823676f, 0.0f, -0.38823676f, 0.8517672f, -0.22352648f }};
        std::array<float32_t, inputData.size()> outputData{};
        LOG_INF("Performing benchmark 2");
        auto duration = Benchmark::benchmark(cycCounter, [&filter, &outputData](const InputT& input) {
            for(size_t iSample = 0; iSample < input.size(); ++iSample)
            {
                outputData[iSample] = filter(input[iSample]);
            }
        }, inputData);
        auto ticks = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        auto time = 1000000U * ticks / CycleCounter::period::den;
        LOG_INF("Benchmark 2 done");
        LOG_INF("Execution time");
        LOG_INF("Ticks = %lld", ticks);
        LOG_INF("Time = %lld us", time);
        logArray(inputData, "Input data (float32)");
        logArray(outputData, "Output data (float32)");
    }

    {
        // sig.butter(1, [3, 12], btype='bandpass', fs=fs, output='sos')
        Dsp::IIRFilter<2> filter{{ 0.38823676f, 0.0f, -0.38823676f, 0.8517672f, -0.22352648f }};
        std::array<float32_t, inputData.size()> outputData{};
        LOG_INF("Performing benchmark 3");
        auto duration = Benchmark::benchmark(cycCounter, [&filter, &outputData](const InputT& input) {
            filter.apply(input, outputData);
        }, inputData);
        auto ticks = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        auto time = 1000000U * ticks / CycleCounter::period::den;
        LOG_INF("Benchmark 3 done");
        LOG_INF("Execution time");
        LOG_INF("Ticks = %lld", ticks);
        LOG_INF("Time = %lld us", time);
        logArray(inputData, "Input data (float32)");
        logArray(outputData, "Output data (float32)");
    }
    

    while(true) { k_msleep(100); }

    return 0;
}
