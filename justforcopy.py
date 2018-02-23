# -*- coding: utf-8 -*-
import requests
import os
from http.cookiejar import LWPCookieJar
from bs4 import BeautifulSoup

def login(logindata,headers):
    loginUrl = r'http://www.qixin.com/api/user/login/'#fiddler抓包发现的真正登录url
    s = requests.session()
    s.cookies = LWPCookieJar('cookiejar')
    if not os.path.exists('cookiejar'):
        lg = s.post(loginUrl, data = logindata, headers = headers)
        s.cookies.save(ignore_discard = True)
    else:
        print('here')
        s.cookies.load(ignore_discard = True)
    return s.cookies



if __name__ == '__main__':
    login_data = {"acc":"18811789859","pass":"qx123"}#网站申请的账户和密码
    m_headers = {
            "Host":"www.qixin.com",
            "Referer":"http://www.qixin.com/auth/login?return_url=%2F",
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0",
    }
    #logout(m_headers)
    cks = login(login_data,m_headers)
    print(cks)
    rootUrl = r"http://www.qixin.com/search?key=京剧团&page="
    pageCount = 1
    #while pageCount < 58:
   # cks ='cookieShowLoginTip=1; sid=s%3AvwkOnRuYiLlHFtBM5uvgxxnWI6O5Fw5S.KdHZ%2BFHo1ZZs7rRI2aVhsZOtCXclkJ4JLCw5NYUHeYo; responseTimeline=36; _zg=%7B%22uuid%22%3A%20%2216062ea0727b1d-086351a53d9487-173c6d56-13c680-16062ea07288ba%22%2C%22sid%22%3A%201513487992.617%2C%22updated%22%3A%201513488619.522%2C%22info%22%3A%201513487992621%2C%22cuid%22%3A%20%221b77dc24-14ae-4ea5-b2a4-257744376df9%22%7D; showsale=1; Hm_lvt_52d64b8d3f6d42a2e416d59635df3f71=1513487993; Hm_lpvt_52d64b8d3f6d42a2e416d59635df3f71=1513488620'
    searchLink = rootUrl + str(pageCount)#登录后搜索“京剧团”，第一页的url就是这个
    searchList = requests.get(searchLink, cookies = cks, headers = m_headers)
    print(searchList.text)