import tushare as ts


ts.set_token('c75d66a12f099b7ced441563e83234d3b73acf437f532a6759a17f10')

pro = ts.pro_api()
pro = ts.pro_api('c75d66a12f099b7ced441563e83234d3b73acf437f532a6759a17f10')



df = pro.trade_cal(exchange='', start_date='20180901', end_date='20181001', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')

# df = pro.query('trade_cal', exchange='', start_date='20180901', end_date='20181001', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')









