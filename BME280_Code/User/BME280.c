#include "BME280.h"

/* Global t_fine used by both temperature and humidity compensation */
int32_t t_fine;

/* Calibration coefficients */
struct _BME280_Calib
{
	uint16_t T1;
	int16_t  T2;
	int16_t  T3;
	uint16_t P1;
	int16_t  P2;
	int16_t  P3;
	int16_t  P4;
	int16_t  P5;
	int16_t  P6;
	int16_t  P7;
	int16_t  P8;
	int16_t  P9;
	uint8_t  H1;
	int16_t  H2;
	uint8_t  H3;
	int16_t  H4;
	int16_t  H5;
	int8_t   H6;
} BME280;

void BME280_Init(GPIO_TypeDef* GPIOx, uint32_t SCL, uint32_t SDA)
{
	uint8_t reg_val, reg_E5;

	SI2C_Init(GPIOx, SCL, SDA, BME280_ADDRESS);

	/* Read temperature calibration */
	BME280.T1 = ((uint16_t)SI2C_ReadReg(BME280_DIG_T1_MSB) << 8) | SI2C_ReadReg(BME280_DIG_T1_LSB);
	BME280.T2 = ((uint16_t)SI2C_ReadReg(BME280_DIG_T2_MSB) << 8) | SI2C_ReadReg(BME280_DIG_T2_LSB);
	BME280.T3 = ((uint16_t)SI2C_ReadReg(BME280_DIG_T3_MSB) << 8) | SI2C_ReadReg(BME280_DIG_T3_LSB);

	/* Read pressure calibration */
	BME280.P1 = ((uint16_t)SI2C_ReadReg(BME280_DIG_P1_MSB) << 8) | SI2C_ReadReg(BME280_DIG_P1_LSB);
	BME280.P2 = ((uint16_t)SI2C_ReadReg(BME280_DIG_P2_MSB) << 8) | SI2C_ReadReg(BME280_DIG_P2_LSB);
	BME280.P3 = ((uint16_t)SI2C_ReadReg(BME280_DIG_P3_MSB) << 8) | SI2C_ReadReg(BME280_DIG_P3_LSB);
	BME280.P4 = ((uint16_t)SI2C_ReadReg(BME280_DIG_P4_MSB) << 8) | SI2C_ReadReg(BME280_DIG_P4_LSB);
	BME280.P5 = ((uint16_t)SI2C_ReadReg(BME280_DIG_P5_MSB) << 8) | SI2C_ReadReg(BME280_DIG_P5_LSB);
	BME280.P6 = ((uint16_t)SI2C_ReadReg(BME280_DIG_P6_MSB) << 8) | SI2C_ReadReg(BME280_DIG_P6_LSB);
	BME280.P7 = ((uint16_t)SI2C_ReadReg(BME280_DIG_P7_MSB) << 8) | SI2C_ReadReg(BME280_DIG_P7_LSB);
	BME280.P8 = ((uint16_t)SI2C_ReadReg(BME280_DIG_P8_MSB) << 8) | SI2C_ReadReg(BME280_DIG_P8_LSB);
	BME280.P9 = ((uint16_t)SI2C_ReadReg(BME280_DIG_P9_MSB) << 8) | SI2C_ReadReg(BME280_DIG_P9_LSB);

	/* Read humidity calibration */
	BME280.H1 = SI2C_ReadReg(BME280_DIG_H1);
	BME280.H2 = ((uint16_t)SI2C_ReadReg(BME280_DIG_H2_MSB) << 8) | SI2C_ReadReg(BME280_DIG_H2_LSB);
	BME280.H3 = SI2C_ReadReg(BME280_DIG_H3);

	/* H4 and H5 are packed across 3 registers (12-bit signed) */
	reg_E5 = SI2C_ReadReg(BME280_DIG_H4_H5_SHARED);
	{
		uint16_t h4_raw = ((uint16_t)SI2C_ReadReg(BME280_DIG_H4_MSB) << 4) | (reg_E5 & 0x0F);
		uint16_t h5_raw = ((uint16_t)SI2C_ReadReg(BME280_DIG_H5_MSB) << 4) | ((reg_E5 >> 4) & 0x0F);
		if (h4_raw > 0x07FF) h4_raw -= 0x1000;
		if (h5_raw > 0x07FF) h5_raw -= 0x1000;
		BME280.H4 = (int16_t)h4_raw;
		BME280.H5 = (int16_t)h5_raw;
	}
	BME280.H6 = (int8_t)SI2C_ReadReg(BME280_DIG_H6);

	/* Configure humidity oversampling x1 */
	SI2C_WriteReg(BME280_REG_CTRL_HUM, OSRS_H_X1);

	/* Configure temperature x1, pressure x4, normal mode */
	reg_val = OSRS_T_X1 | OSRS_P_X4 | MODE_NORMAL;
	SI2C_WriteReg(BME280_REG_CTRL_MEAS, reg_val);

	/* Configure standby 250ms, filter x4 */
	reg_val = T_SB_250MS | FILTER_4;
	SI2C_WriteReg(BME280_REG_CONFIG, reg_val);
}

uint8_t BME280_Get_ID(void)
{
	return SI2C_ReadReg(BME280_REG_ID);
}

double BME280_Get_Temp(void)
{
	int32_t adc_T;
	uint8_t msb, lsb, xlsb;

	msb  = SI2C_ReadReg(BME280_REG_TEMP_MSB);
	lsb  = SI2C_ReadReg(BME280_REG_TEMP_LSB);
	xlsb = SI2C_ReadReg(BME280_REG_TEMP_XLSB);

	adc_T = ((int32_t)msb << 12) | ((int32_t)lsb << 4) | (xlsb >> 4);

	double var1, var2, T;
	var1 = (((double)adc_T) / 16384.0 - ((double)BME280.T1) / 1024.0) * ((double)BME280.T2);
	var2 = ((((double)adc_T) / 131072.0 - ((double)BME280.T1) / 8192.0) *
	        (((double)adc_T) / 131072.0 - ((double)BME280.T1) / 8192.0)) * ((double)BME280.T3);
	t_fine = (int32_t)(var1 + var2);
	T = (var1 + var2) / 5120.0;
	return T;
}

double BME280_Get_Press(void)
{
	int32_t adc_P;
	uint8_t msb, lsb, xlsb;

	msb  = SI2C_ReadReg(BME280_REG_PRESS_MSB);
	lsb  = SI2C_ReadReg(BME280_REG_PRESS_LSB);
	xlsb = SI2C_ReadReg(BME280_REG_PRESS_XLSB);

	adc_P = ((int32_t)msb << 12) | ((int32_t)lsb << 4) | (xlsb >> 4);

	double var1, var2, p;
	var1 = ((double)t_fine / 2.0) - 64000.0;
	var2 = var1 * var1 * ((double)BME280.P6) / 32768.0;
	var2 = var2 + var1 * ((double)BME280.P5) * 2.0;
	var2 = (var2 / 4.0) + (((double)BME280.P4) * 65536.0);
	var1 = (((double)BME280.P3) * var1 * var1 / 524288.0 + ((double)BME280.P2) * var1) / 524288.0;
	var1 = (1.0 + var1 / 32768.0) * ((double)BME280.P1);
	if (var1 == 0.0) return 0.0;
	p = 1048576.0 - (double)adc_P;
	p = (p - (var2 / 4096.0)) * 6250.0 / var1;
	var1 = ((double)BME280.P9) * p * p / 2147483648.0;
	var2 = p * ((double)BME280.P8) / 32768.0;
	p = p + (var1 + var2 + ((double)BME280.P7)) / 16.0;
	return p;
}

double BME280_Get_Hum(void)
{
	int32_t adc_H;
	uint8_t msb, lsb;

	msb = SI2C_ReadReg(BME280_REG_HUM_MSB);
	lsb = SI2C_ReadReg(BME280_REG_HUM_LSB);

	adc_H = ((int32_t)msb << 8) | lsb;

	double var_H;
	var_H = (((double)t_fine) - 76800.0);
	var_H = ((double)adc_H - (((double)BME280.H4) * 64.0 + ((double)BME280.H5) / 16384.0 * var_H)) *
	        (((double)BME280.H2) / 65536.0 * (1.0 + ((double)BME280.H6) / 67108864.0 * var_H *
	        (1.0 + ((double)BME280.H3) / 67108864.0 * var_H)));
	var_H = var_H * (1.0 - ((double)BME280.H1) * var_H / 524288.0);
	if (var_H > 100.0) var_H = 100.0;
	if (var_H < 0.0)   var_H = 0.0;
	return var_H;
}
