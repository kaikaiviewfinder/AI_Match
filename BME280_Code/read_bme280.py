import serial
import re
import datetime

# ====== 配置 ======
PORT = 'COM3'        # 改成你电脑上的实际串口号
BAUD = 115200
TIMEOUT = 2
SEA_LEVEL_PRESSURE = 1013.25   # 海平面参考气压 (hPa)，可用当地气象站 QNH 修正
# ==================

def pressure_to_altitude(pressure_hpa, sea_level_hpa=1013.25, temperature_c=15.0):
    """气压 → 海拔 (ISA 标准大气模型)"""
    T0 = temperature_c + 273.15
    return (T0 / 0.0065) * (1 - (pressure_hpa / sea_level_hpa) ** 0.190263)


def main():
    ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    print(f"打开串口 {PORT}, 波特率 {BAUD}")
    print("等待数据...\n")

    # 正则匹配: Temperature: XX.XX C  |  Humidity: XX.XX %  |  Pressure: XX.XX Pa
    pattern = re.compile(
        r"Temperature:\s*(?P<temp>[\d.]+)\s*C\s*\|"
        r"\s*Humidity:\s*(?P<hum>[\d.]+)\s*%\s*\|"
        r"\s*Pressure:\s*(?P<pres>[\d.]+)\s*Pa"
    )

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            match = pattern.search(line)
            if match:
                temp = float(match.group('temp'))
                hum  = float(match.group('hum'))
                pres = float(match.group('pres')) / 100.0  # Pa -> hPa

                altitude = pressure_to_altitude(pres, SEA_LEVEL_PRESSURE, temp)

                now = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"[{now}] 温度: {temp:.2f} °C  |  湿度: {hum:.2f} %  |  气压: {pres:.2f} hPa  |  海拔: {altitude:.2f} m")
            else:
                # 非数据行也打印出来 (如启动信息、错误等)
                if line:
                    print(line)

    except KeyboardInterrupt:
        print("\n用户停止")
    finally:
        ser.close()
        print("串口已关闭")


if __name__ == '__main__':
    main()
