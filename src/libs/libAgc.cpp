#include "common/abi.h"
#include "libs/agc.h"
#include "libs/libs.h"
#include "loader/symbolDatabase.h"

#include <atomic>
#include <cstdio>

namespace Libs {

namespace LibAgc {

LIB_VERSION("Agc", 1, "Agc", 1, 1);

namespace Gen5 = Graphics::Gen5;

// Soft-stubs for TLOU Agc_v1 NIDs not yet mapped to Graphics5 implementations.
#define KYTY_AGC_STUB(fn, nid_str)                                                                 \
	static KYTY_SYSV_ABI uint64_t fn(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,            \
	                                 uint64_t a4, uint64_t a5) {                                    \
		(void)a0;                                                                                  \
		(void)a1;                                                                                  \
		(void)a2;                                                                                  \
		(void)a3;                                                                                  \
		(void)a4;                                                                                  \
		(void)a5;                                                                                  \
		static std::atomic<uint32_t> call_count {0};                                               \
		const auto                   n = call_count.fetch_add(1, std::memory_order_relaxed);       \
		LOGF("AgcStub: NID=%s call=%u\n", nid_str, n);                                             \
		if (n < 64) {                                                                              \
			fprintf(stderr, "AgcStub: NID=%s call=%u\n", nid_str, n);                              \
		}                                                                                          \
		return 0;                                                                                  \
	}
KYTY_AGC_STUB(AgcStub__u6dKSLWM2o, "+u6dKSLWM2o")
KYTY_AGC_STUB(AgcStub_N_03RZmELWWzw, "03RZmELWWzw")
KYTY_AGC_STUB(AgcStub_N_0o3VDdtA6nM, "0o3VDdtA6nM")
KYTY_AGC_STUB(AgcStub_N_0ZOG0jc9nRg, "0ZOG0jc9nRg")
KYTY_AGC_STUB(AgcStub_N_1_gUn1PI4Sw, "1-gUn1PI4Sw")
KYTY_AGC_STUB(AgcStub_N_1tB0xkLNjcw, "1tB0xkLNjcw")
KYTY_AGC_STUB(AgcStub_N_2ccJz9LQI_w, "2ccJz9LQI+w")
KYTY_AGC_STUB(AgcStub_N_7toV_elXqNM, "7toV+elXqNM")
KYTY_AGC_STUB(AgcStub_N_7Wa3aeJgeVU, "7Wa3aeJgeVU")
KYTY_AGC_STUB(AgcStub_N_9S4noWrUI0s, "9S4noWrUI0s")
KYTY_AGC_STUB(AgcStub_AAeX_U5_P3M, "AAeX-U5-P3M")
KYTY_AGC_STUB(AgcStub_AFIh8SQkYlQ, "AFIh8SQkYlQ")
KYTY_AGC_STUB(AgcStub_aP1Ki9G3__4, "aP1Ki9G3++4")
KYTY_AGC_STUB(AgcStub_b5u0Jzm8TF8, "b5u0Jzm8TF8")
KYTY_AGC_STUB(AgcStub_b_oySn_G2tE, "b-oySn+G2tE")
KYTY_AGC_STUB(AgcStub_C4l9fB17t8w, "C4l9fB17t8w")
KYTY_AGC_STUB(AgcStub_ca4KPvp0qLQ, "ca4KPvp0qLQ")
KYTY_AGC_STUB(AgcStub_CbQh3DKMSno, "CbQh3DKMSno")
KYTY_AGC_STUB(AgcStub_da1Sm8_QDoU, "da1Sm8-QDoU")
KYTY_AGC_STUB(AgcStub_DwICrVxerkY, "DwICrVxerkY")
KYTY_AGC_STUB(AgcStub_e1DFTg_Sd8U, "e1DFTg+Sd8U")
KYTY_AGC_STUB(AgcStub_ebixW91gpPw, "ebixW91gpPw")
KYTY_AGC_STUB(AgcStub_eCjKaqeeQ5s, "eCjKaqeeQ5s")
KYTY_AGC_STUB(AgcStub_eWaWyFegzgQ, "eWaWyFegzgQ")
KYTY_AGC_STUB(AgcStub_F8NLhWvFemI, "F8NLhWvFemI")
KYTY_AGC_STUB(AgcStub_FcgdDM3MB_k, "FcgdDM3MB+k")
KYTY_AGC_STUB(AgcStub_FneFypEDRgY, "FneFypEDRgY")
KYTY_AGC_STUB(AgcStub_FuVbkyKlf_s, "FuVbkyKlf+s")
KYTY_AGC_STUB(AgcStub_G0jrLdvEqDw, "G0jrLdvEqDw")
KYTY_AGC_STUB(AgcStub_GBCh3zCihoU, "GBCh3zCihoU")
KYTY_AGC_STUB(AgcStub_GPbUp9jXQa8, "GPbUp9jXQa8")
KYTY_AGC_STUB(AgcStub_gQkqkLttcpw, "gQkqkLttcpw")
KYTY_AGC_STUB(AgcStub_GXBlM_ekzrI, "GXBlM-ekzrI")
KYTY_AGC_STUB(AgcStub_H6vHS5cidSA, "H6vHS5cidSA")
KYTY_AGC_STUB(AgcStub_hcIxS8pmXF4, "hcIxS8pmXF4")
KYTY_AGC_STUB(AgcStub_idlaArvdXEs, "idlaArvdXEs")
KYTY_AGC_STUB(AgcStub_j4emHHndCPY, "j4emHHndCPY")
KYTY_AGC_STUB(AgcStub_J8YCgfKAMQs, "J8YCgfKAMQs")
KYTY_AGC_STUB(AgcStub_JOWmDrl_j20, "JOWmDrl+j20")
KYTY_AGC_STUB(AgcStub_jt3pl7EN17o, "jt3pl7EN17o")
KYTY_AGC_STUB(AgcStub_k0E7vkgqAuE, "k0E7vkgqAuE")
KYTY_AGC_STUB(AgcStub_K2mciNVxUCE, "K2mciNVxUCE")
KYTY_AGC_STUB(AgcStub_KjPeVduz6jU, "KjPeVduz6jU")
KYTY_AGC_STUB(AgcStub_kUlvghKs_mA, "kUlvghKs-mA")
KYTY_AGC_STUB(AgcStub_M0ttm8h7SKA, "M0ttm8h7SKA")
KYTY_AGC_STUB(AgcStub_MDLD5Ly94Xk, "MDLD5Ly94Xk")
KYTY_AGC_STUB(AgcStub_mljzuGDZRQ4, "mljzuGDZRQ4")
KYTY_AGC_STUB(AgcStub_MMlmJAL7N5w, "MMlmJAL7N5w")
KYTY_AGC_STUB(AgcStub_ms1xVoZ_Vwc, "ms1xVoZ-Vwc")
KYTY_AGC_STUB(AgcStub_mStuvI0zOtc, "mStuvI0zOtc")
KYTY_AGC_STUB(AgcStub_n485EBnIWmk, "n485EBnIWmk")
KYTY_AGC_STUB(AgcStub_NKIzURsgV7I, "NKIzURsgV7I")
KYTY_AGC_STUB(AgcStub_nNlUtdDDvZ0, "nNlUtdDDvZ0")
KYTY_AGC_STUB(AgcStub_opR1JeJZCBU, "opR1JeJZCBU")
KYTY_AGC_STUB(AgcStub_oz6zQq1JwCE, "oz6zQq1JwCE")
KYTY_AGC_STUB(AgcStub_P1CugZ99Uzc, "P1CugZ99Uzc")
KYTY_AGC_STUB(AgcStub_PxKWV2fVAps, "PxKWV2fVAps")
KYTY_AGC_STUB(AgcStub_pYoKs3lPy88, "pYoKs3lPy88")
KYTY_AGC_STUB(AgcStub_q4VuU_QsLOE, "q4VuU-QsLOE")
KYTY_AGC_STUB(AgcStub_QhPDD513V0w, "QhPDD513V0w")
KYTY_AGC_STUB(AgcStub_r98I08t_LOg, "r98I08t+LOg")
KYTY_AGC_STUB(AgcStub_rP5xLdOf26k, "rP5xLdOf26k")
KYTY_AGC_STUB(AgcStub_rUuVjyR_Rd4, "rUuVjyR+Rd4")
KYTY_AGC_STUB(AgcStub_rVOmPz2RBlg, "rVOmPz2RBlg")
KYTY_AGC_STUB(AgcStub_szG7hz2yEhA, "szG7hz2yEhA")
KYTY_AGC_STUB(AgcStub_T9fjQIINoeE, "T9fjQIINoeE")
KYTY_AGC_STUB(AgcStub_TGEZzUWLbrc, "TGEZzUWLbrc")
KYTY_AGC_STUB(AgcStub_UQGTw4xRlcM, "UQGTw4xRlcM")
KYTY_AGC_STUB(AgcStub_uZW_mqsxkrM, "uZW-mqsxkrM")
KYTY_AGC_STUB(AgcStub_vLrBL8DQiz8, "vLrBL8DQiz8")
KYTY_AGC_STUB(AgcStub_XKKuA6VkSRc, "XKKuA6VkSRc")
KYTY_AGC_STUB(AgcStub_XN_Iuu7XsM8, "XN+Iuu7XsM8")
KYTY_AGC_STUB(AgcStub_Y_5vneiBtzk, "Y-5vneiBtzk")
KYTY_AGC_STUB(AgcStub_yheJGN_ay_A, "yheJGN-ay+A")
KYTY_AGC_STUB(AgcStub_yUBESvCCJ4I, "yUBESvCCJ4I")
KYTY_AGC_STUB(AgcStub_zARR5aCmkoY, "zARR5aCmkoY")
KYTY_AGC_STUB(AgcStub_zg6u_N6Otxs, "zg6u-N6Otxs")
KYTY_AGC_STUB(AgcStub_ziVA3whp3p4, "ziVA3whp3p4")

#undef KYTY_AGC_STUB

LIB_DEFINE(InitAgc_1) {
	PRINT_NAME_ENABLE(true);

	// Alias core Graphics5 builders under Agc (same NIDs) when available.
	LIB_FUNC("23LRUSvYu1M", Gen5::GraphicsInit);
	LIB_FUNC("YUeqkyT7mEQ", Gen5::GraphicsDcbSetFlip);

	LIB_FUNC("+u6dKSLWM2o", AgcStub__u6dKSLWM2o);
	LIB_FUNC("03RZmELWWzw", AgcStub_N_03RZmELWWzw);
	LIB_FUNC("0o3VDdtA6nM", AgcStub_N_0o3VDdtA6nM);
	LIB_FUNC("0ZOG0jc9nRg", AgcStub_N_0ZOG0jc9nRg);
	LIB_FUNC("1-gUn1PI4Sw", AgcStub_N_1_gUn1PI4Sw);
	LIB_FUNC("1tB0xkLNjcw", AgcStub_N_1tB0xkLNjcw);
	LIB_FUNC("2ccJz9LQI+w", AgcStub_N_2ccJz9LQI_w);
	LIB_FUNC("7toV+elXqNM", AgcStub_N_7toV_elXqNM);
	LIB_FUNC("7Wa3aeJgeVU", AgcStub_N_7Wa3aeJgeVU);
	LIB_FUNC("9S4noWrUI0s", AgcStub_N_9S4noWrUI0s);
	LIB_FUNC("AAeX-U5-P3M", AgcStub_AAeX_U5_P3M);
	LIB_FUNC("AFIh8SQkYlQ", AgcStub_AFIh8SQkYlQ);
	LIB_FUNC("aP1Ki9G3++4", AgcStub_aP1Ki9G3__4);
	LIB_FUNC("b5u0Jzm8TF8", AgcStub_b5u0Jzm8TF8);
	LIB_FUNC("b-oySn+G2tE", AgcStub_b_oySn_G2tE);
	LIB_FUNC("C4l9fB17t8w", AgcStub_C4l9fB17t8w);
	LIB_FUNC("ca4KPvp0qLQ", AgcStub_ca4KPvp0qLQ);
	LIB_FUNC("CbQh3DKMSno", AgcStub_CbQh3DKMSno);
	LIB_FUNC("da1Sm8-QDoU", AgcStub_da1Sm8_QDoU);
	LIB_FUNC("DwICrVxerkY", AgcStub_DwICrVxerkY);
	LIB_FUNC("e1DFTg+Sd8U", AgcStub_e1DFTg_Sd8U);
	LIB_FUNC("ebixW91gpPw", AgcStub_ebixW91gpPw);
	LIB_FUNC("eCjKaqeeQ5s", AgcStub_eCjKaqeeQ5s);
	LIB_FUNC("eWaWyFegzgQ", AgcStub_eWaWyFegzgQ);
	LIB_FUNC("F8NLhWvFemI", AgcStub_F8NLhWvFemI);
	LIB_FUNC("FcgdDM3MB+k", AgcStub_FcgdDM3MB_k);
	LIB_FUNC("FneFypEDRgY", AgcStub_FneFypEDRgY);
	LIB_FUNC("FuVbkyKlf+s", AgcStub_FuVbkyKlf_s);
	LIB_FUNC("G0jrLdvEqDw", AgcStub_G0jrLdvEqDw);
	LIB_FUNC("GBCh3zCihoU", AgcStub_GBCh3zCihoU);
	LIB_FUNC("GPbUp9jXQa8", AgcStub_GPbUp9jXQa8);
	LIB_FUNC("gQkqkLttcpw", AgcStub_gQkqkLttcpw);
	LIB_FUNC("GXBlM-ekzrI", AgcStub_GXBlM_ekzrI);
	LIB_FUNC("H6vHS5cidSA", AgcStub_H6vHS5cidSA);
	LIB_FUNC("hcIxS8pmXF4", AgcStub_hcIxS8pmXF4);
	LIB_FUNC("idlaArvdXEs", AgcStub_idlaArvdXEs);
	LIB_FUNC("j4emHHndCPY", AgcStub_j4emHHndCPY);
	LIB_FUNC("J8YCgfKAMQs", AgcStub_J8YCgfKAMQs);
	LIB_FUNC("JOWmDrl+j20", AgcStub_JOWmDrl_j20);
	LIB_FUNC("jt3pl7EN17o", AgcStub_jt3pl7EN17o);
	LIB_FUNC("k0E7vkgqAuE", AgcStub_k0E7vkgqAuE);
	LIB_FUNC("K2mciNVxUCE", AgcStub_K2mciNVxUCE);
	LIB_FUNC("KjPeVduz6jU", AgcStub_KjPeVduz6jU);
	LIB_FUNC("kUlvghKs-mA", AgcStub_kUlvghKs_mA);
	LIB_FUNC("M0ttm8h7SKA", AgcStub_M0ttm8h7SKA);
	LIB_FUNC("MDLD5Ly94Xk", AgcStub_MDLD5Ly94Xk);
	LIB_FUNC("mljzuGDZRQ4", AgcStub_mljzuGDZRQ4);
	LIB_FUNC("MMlmJAL7N5w", AgcStub_MMlmJAL7N5w);
	LIB_FUNC("ms1xVoZ-Vwc", AgcStub_ms1xVoZ_Vwc);
	LIB_FUNC("mStuvI0zOtc", AgcStub_mStuvI0zOtc);
	LIB_FUNC("n485EBnIWmk", AgcStub_n485EBnIWmk);
	LIB_FUNC("NKIzURsgV7I", AgcStub_NKIzURsgV7I);
	LIB_FUNC("nNlUtdDDvZ0", AgcStub_nNlUtdDDvZ0);
	LIB_FUNC("opR1JeJZCBU", AgcStub_opR1JeJZCBU);
	LIB_FUNC("oz6zQq1JwCE", AgcStub_oz6zQq1JwCE);
	LIB_FUNC("P1CugZ99Uzc", AgcStub_P1CugZ99Uzc);
	LIB_FUNC("PxKWV2fVAps", AgcStub_PxKWV2fVAps);
	LIB_FUNC("pYoKs3lPy88", AgcStub_pYoKs3lPy88);
	LIB_FUNC("q4VuU-QsLOE", AgcStub_q4VuU_QsLOE);
	LIB_FUNC("QhPDD513V0w", AgcStub_QhPDD513V0w);
	LIB_FUNC("r98I08t+LOg", AgcStub_r98I08t_LOg);
	LIB_FUNC("rP5xLdOf26k", AgcStub_rP5xLdOf26k);
	LIB_FUNC("rUuVjyR+Rd4", AgcStub_rUuVjyR_Rd4);
	LIB_FUNC("rVOmPz2RBlg", AgcStub_rVOmPz2RBlg);
	LIB_FUNC("szG7hz2yEhA", AgcStub_szG7hz2yEhA);
	LIB_FUNC("T9fjQIINoeE", AgcStub_T9fjQIINoeE);
	LIB_FUNC("TGEZzUWLbrc", AgcStub_TGEZzUWLbrc);
	LIB_FUNC("UQGTw4xRlcM", AgcStub_UQGTw4xRlcM);
	LIB_FUNC("uZW-mqsxkrM", AgcStub_uZW_mqsxkrM);
	LIB_FUNC("vLrBL8DQiz8", AgcStub_vLrBL8DQiz8);
	LIB_FUNC("XKKuA6VkSRc", AgcStub_XKKuA6VkSRc);
	LIB_FUNC("XN+Iuu7XsM8", AgcStub_XN_Iuu7XsM8);
	LIB_FUNC("Y-5vneiBtzk", AgcStub_Y_5vneiBtzk);
	LIB_FUNC("yheJGN-ay+A", AgcStub_yheJGN_ay_A);
	LIB_FUNC("yUBESvCCJ4I", AgcStub_yUBESvCCJ4I);
	LIB_FUNC("zARR5aCmkoY", AgcStub_zARR5aCmkoY);
	LIB_FUNC("zg6u-N6Otxs", AgcStub_zg6u_N6Otxs);
	LIB_FUNC("ziVA3whp3p4", AgcStub_ziVA3whp3p4);

}

} // namespace LibAgc

} // namespace Libs
