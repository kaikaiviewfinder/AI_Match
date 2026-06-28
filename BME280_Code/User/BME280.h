#ifndef _BME280_H
#define _BME280_H

#include "stm32f10x.h"
#include "MyI2C.h"

/* BME280 I2C address: SDO=GND -> 0x76, SDO=VCC -> 0x77 */
#define BME280_ADDRESS          0x76

/* Register map */
#define BME280_REG_ID           0xD0
#define BME280_REG_RESET        0xE0
#define BME280_REG_CTRL_HUM     0xF2
#define BME280_REG_STATUS       0xF3
#define BME280_REG_CTRL_MEAS    0xF4
#define BME280_REG_CONFIG       0xF5
#define BME280_REG_PRESS_MSB    0xF7
#define BME280_REG_PRESS_LSB    0xF8
#define BME280_REG_PRESS_XLSB   0xF9
#define BME280_REG_TEMP_MSB     0xFA
#define BME280_REG_TEMP_LSB     0xFB
#define BME280_REG_TEMP_XLSB    0xFC
#define BME280_REG_HUM_MSB      0xFD
#define BME280_REG_HUM_LSB      0xFE

/* Calibration registers */
#define BME280_DIG_T1_LSB       0x88
#define BME280_DIG_T1_MSB       0x89
#define BME280_DIG_T2_LSB       0x8A
#define BME280_DIG_T2_MSB       0x8B
#define BME280_DIG_T3_LSB       0x8C
#define BME280_DIG_T3_MSB       0x8D
#define BME280_DIG_P1_LSB       0x8E
#define BME280_DIG_P1_MSB       0x8F
#define BME280_DIG_P2_LSB       0x90
#define BME280_DIG_P2_MSB       0x91
#define BME280_DIG_P3_LSB       0x92
#define BME280_DIG_P3_MSB       0x93
#define BME280_DIG_P4_LSB       0x94
#define BME280_DIG_P4_MSB       0x95
#define BME280_DIG_P5_LSB       0x96
#define BME280_DIG_P5_MSB       0x97
#define BME280_DIG_P6_LSB       0x98
#define BME280_DIG_P6_MSB       0x99
#define BME280_DIG_P7_LSB       0x9A
#define BME280_DIG_P7_MSB       0x9B
#define BME280_DIG_P8_LSB       0x9C
#define BME280_DIG_P8_MSB       0x9D
#define BME280_DIG_P9_LSB       0x9E
#define BME280_DIG_P9_MSB       0x9F
#define BME280_DIG_H1           0xA1
#define BME280_DIG_H2_LSB       0xE1
#define BME280_DIG_H2_MSB       0xE2
#define BME280_DIG_H3           0xE3
#define BME280_DIG_H4_MSB       0xE4
#define BME280_DIG_H4_H5_SHARED 0xE5
#define BME280_DIG_H5_MSB       0xE6
#define BME280_DIG_H6           0xE7

/* Oversampling options */
#define OSRS_T_SKIP  0x00
#define OSRS_T_X1    0x20
#define OSRS_T_X2    0x40
#define OSRS_T_X4    0x60
#define OSRS_T_X8    0x80
#define OSRS_T_X16   0xE0

#define OSRS_P_SKIP  0x00
#define OSRS_P_X1    0x04
#define OSRS_P_X2    0x08
#define OSRS_P_X4    0x0C
#define OSRS_P_X8    0x10
#define OSRS_P_X16   0x1C

#define OSRS_H_SKIP  0x00
#define OSRS_H_X1    0x01
#define OSRS_H_X2    0x02
#define OSRS_H_X4    0x03
#define OSRS_H_X8    0x04
#define OSRS_H_X16   0x05

/* Mode */
#define MODE_SLEEP   0x00
#define MODE_FORCED  0x01
#define MODE_NORMAL  0x03

/* Standby time */
#define T_SB_0_5MS   0x00
#define T_SB_62_5MS  0x20
#define T_SB_125MS   0x40
#define T_SB_250MS   0x60
#define T_SB_500MS   0x80
#define T_SB_1000MS  0xA0
#define T_SB_2000MS  0xC0
#define T_SB_4000MS  0xE0

/* Filter */
#define FILTER_OFF   0x00
#define FILTER_2     0x04
#define FILTER_4     0x08
#define FILTER_8     0x0C
#define FILTER_16    0x1C

void BME280_Init(GPIO_TypeDef* GPIOx, uint32_t SCL, uint32_t SDA);
uint8_t BME280_Get_ID(void);
double BME280_Get_Temp(void);
double BME280_Get_Press(void);
double BME280_Get_Hum(void);

#endif
