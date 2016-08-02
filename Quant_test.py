#coding:utf-8

from WindPy import *
import time

def myWSQCallback(indata):
    """
    作者：朱洪海  时间20130713
    DemoWSQCallback 是WSQ订阅时提供的回调函数模板。该函数只有一个为w.WindData类型的参数indata。
    该函数是被C中线程调用的，因此此线程应该仅仅限于简单的数据处理，并且还应该主要线程之间互斥考虑。

    用户自定义回调函数，请一定要使用try...except
    """
    try:
        # print(type(indata))
        print(indata)
        print(time.mktime((indata.Times[0].timetuple())))
        print str(indata.Fields[0]), ':', indata.Data[0][0]
        print str(indata.Fields[1]), ':', indata.Data[1][0]

    except Exception as e:
        print(e)
        return

########################################################################################
if __name__=='__main__':
    try:
        w.start() # 启动Wind API
    except Exception as e:
        print e
        exit(2)

    # data=w.wsq("RB1610.SHF","rt_last, rt_last_vol, rt_chg",func=myWSQCallback)
    data=w.wsq("RB1610.SHF","rt_pre_close,rt_open,rt_high,rt_low,rt_last,rt_last_amt,rt_last_vol,rt_latest,rt_vol,rt_amt,\
rt_chg,rt_pct_chg,rt_high_limit,rt_low_limit,rt_swing,rt_vwap,rt_upward_vol,rt_downward_vol,rt_bsize_total,\
rt_asize_total,rt_vol_ratio,rt_turn,rt_pre_iopv,rt_iopv,rt_mkt_cap,rt_float_mkt_cap,rt_pre_oi,rt_oi,rt_oi_chg,\
rt_pre_settle,rt_settle,rt_discount,rt_discount_ratio,rt_pe_ttm,rt_pb_lf,rt_rise_days,rt_spread,rt_susp_flag,\
rt_last_cp,rt_last_ytm,rt_pre_close_dp,rt_delta,rt_gamma,rt_vega,rt_theta,rt_rho,rt_imp_volatility",func=myWSQCallback)
    print data

    time.sleep(10)
    w.cancelRequest(data.RequestID)
    print w.isconnected()
    # w.stop()


'''
[w_wsq_data,w_wsq_codes,w_wsq_fields,w_wsq_times,w_wsq_errorid,w_wsq_reqid]=
w.wsq('000002.SZ,000004.SZ,000001.SZ,510050.SH,510300.OF',
'rt_date,rt_time,rt_pre_close,rt_open,rt_high,rt_low,rt_last,rt_last_amt,rt_last_vol,rt_latest,rt_vol,rt_amt,
rt_chg,rt_pct_chg,rt_high_limit,rt_low_limit,rt_swing,rt_vwap,rt_upward_vol,rt_downward_vol,rt_bsize_total,
rt_asize_total,rt_vol_ratio,rt_turn,rt_pre_iopv,rt_iopv,rt_mkt_cap,rt_float_mkt_cap,rt_pre_oi,rt_oi,rt_oi_chg,
rt_pre_settle,rt_settle,rt_discount,rt_discount_ratio,rt_pe_ttm,rt_pb_lf,rt_rise_days,rt_spread,rt_susp_flag,
rt_high_52wk,rt_low_52wk,rt_pct_chg_1min,rt_pct_chg_3min,rt_pct_chg_5d,rt_pct_chg_10d,rt_pct_chg_20d,
rt_pct_chg_60d,rt_last_cp,rt_last_ytm,rt_pre_close_dp,rt_ask1,rt_ask2,rt_ask3,rt_ask4,rt_ask5,rt_ask6,rt_ask7,
rt_ask8,rt_ask9,rt_ask10,rt_bid1,rt_bid2,rt_bid3,rt_bid4,rt_bid5,rt_bid6,rt_bid7,rt_bid8,rt_bid9,rt_bid10,rt_bsize1,
rt_bsize2,rt_bsize3,rt_bsize4,rt_bsize5,rt_bsize6,rt_bsize7,rt_bsize8,rt_bsize9,rt_bsize10,rt_asize1,rt_asize2,
rt_asize3,rt_asize4,rt_asize5,rt_asize6,rt_asize7,rt_asize8,rt_asize9,rt_asize10,rt_ma_5d,rt_ma_10d,rt_ma_20d,
rt_ma_60d,rt_ma_120d,rt_ma_250d,rt_delta,rt_gamma,rt_vega,rt_theta,rt_rho,rt_imp_volatility',@wsqcallback)
'''
