'''
docstring
'''
import os
import subprocess  # shell 命令
import threading
import time

PATH = r'./Data'
output = './output.txt'  # 待下载的链接
# 结果输出
timeout_txt = './timeout.txt'
succeed_txt = './succeed.txt'
wrong_txt = './wrong.txt'

Max_process_number = 6  # 子线程个数
trytime = 3  # 重试次数
Timeout = 60  # 超时时间


def readline_txt(output):
    with open(output, 'r') as f:  # 读取连接
        LINK = f.readlines()  # 读取所有行
    read = LINK[-1]  # 读取最后一行
    LINK.pop()  # 移除掉最后一行注释
    print(read + ' check ' + str(len(LINK)))
    return LINK


def writing_txt_line(file_name, contents, Mode=1):  # 输出记录
    if Mode == 0:
        with open(file_name, 'w') as f:
            f.writelines('')
    else:
        with open(file_name, 'a') as f:
            f.writelines(contents + '\n')


def Download_wget_OS(List, PATH):  # url列表下载
    cmd = 'wget -c ' + List + ' -P ' + PATH + ' -t ' + str(trytime) + ' -T ' + str(Timeout)
    # print(cmd+'\n')#打印命令
    status_subprocess = subprocess.call(cmd, shell=True)  # 返回程序状态
    # print('正在下载: '+List)
    # print(status_subprocess)
    file = List.rsplit('/', 1)[-1]
    if status_subprocess == 0:
        print('下载成功:' + file)
        return '下载成功'
    elif status_subprocess == 1:
        print('链接错误:' + file)
        return '链接错误'
    elif status_subprocess == 4:
        print('链接超时:' + file)
        return '链接超时'
    else:
        print('其他错误:', status_subprocess, +file)
    return '其他错误'


class MyThread(threading.Thread):  # 自己写的类，带2个输入参数
    def __init__(self, List, PATH):  # 必须含有__init__方法和run方法
        threading.Thread.__init__(self)  # 初始化 函数进入时都先执行这一块
        self.List = List
        self.PATH = PATH
        self.result = 0

    def run(self):  # 然后进入函数将运行的内容
        self.result = Download_wget_OS(self.List, self.PATH)

    def get_result(self):  # 其它方法 返回结果
        return self.result

    def Name(self):  # 其它方法 返回名字
        return self.List


def Multi_process(List, PATH):  # 并行下载
    threads_group = []  # 线程池
    succeed_count = 0
    linkerror_count = 0
    timeout_count = 0
    other_count = 0
    Total = len(List)
    writing_txt_line(timeout_txt, '', Mode=0)  ##清空文本
    writing_txt_line(succeed_txt, '', Mode=0)
    writing_txt_line(wrong_txt, '', Mode=0)

    for i in range(len(List)):
        url_list = List[i].strip()  # 去掉头尾多余符号
        # 总线程数小于等于Max_process_number 主线程也算一个

        New_threads = MyThread(url_list, PATH)  # 自定义的类
        threads_group.append(New_threads)  # 线程名添加进池里
        New_threads.setDaemon(True)  # 设为守护线程
        New_threads.start()

        print('启动线程:' + url_list.rsplit('/', 1)[-1])
        print('总线程数:', len(threading.enumerate()), ' 下载线程数:', len(threads_group))
        while (len(threading.enumerate()) > Max_process_number):
            # 总线程数小于等于Max_process_number时退出
            pass  # 不操作
        # 检查子线程 并计数
        for threads in threads_group:
            # 计数部分
            if threads.get_result() == '下载成功':
                succeed_count = succeed_count + 1
                writing_txt_line(succeed_txt, threads.Name())  # 写入对应连接
                # 在这里可以加入校验

            elif threads.get_result() == '链接超时':
                timeout_count = timeout_count + 1
                writing_txt_line(timeout_txt, threads.Name())  # 写入对应连接

            elif threads.get_result() == '链接错误':
                linkerror_count = linkerror_count + 1
                writing_txt_line(wrong_txt, threads.Name())  # 写入对应连接

            elif threads.get_result() == '其他错误':
                other_count = other_count + 1
                writing_txt_line(wrong_txt, threads.Name())  # 写入对应连接

            # 移除已完成子线程
            if threads.get_result() != 0:  # 检查哪个子线程执行完
                print('移除线程:' + threads.Name().rsplit('/', 1)[-1])
                threads_group.remove(threads)
                break
            print('总数:', Total, '下载成功:', succeed_count, '链接超时:', timeout_count, '链接错误:', linkerror_count,
                  '其他错误:', other_count)
    threads_group.join()  # 需要子线程阻塞
    print('所有下载已启动')


if __name__ == '__main__':
    # Download_wget_OS(url,PATH)
    Multi_process(readline_txt(output), PATH)
    time.sleep(5)

