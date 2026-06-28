#include "MyI2C.h"

struct I2C_PIN {
	GPIO_TypeDef* GPIOx;
	uint16_t SCL;
	uint16_t SDA;
	uint16_t I2C_Add;
} I2C;

static void MyI2C_W_SCL(uint8_t BitValue)
{
	GPIO_WriteBit(I2C.GPIOx, I2C.SCL, (BitAction)BitValue);
	Delay_us(10);
}

static void MyI2C_W_SDA(uint8_t BitValue)
{
	GPIO_WriteBit(I2C.GPIOx, I2C.SDA, (BitAction)BitValue);
	Delay_us(10);
}

static uint8_t MyI2C_R_SDA(void)
{
	uint8_t BitValue = GPIO_ReadInputDataBit(I2C.GPIOx, I2C.SDA);
	Delay_us(10);
	return BitValue;
}

void SI2C_Init(GPIO_TypeDef* GPIOx, uint16_t SCL, uint16_t SDA, uint16_t I2C_Address)
{
	I2C.SCL = SCL;
	I2C.SDA = SDA;
	I2C.GPIOx = GPIOx;
	I2C.I2C_Add = I2C_Address;

	if      (GPIOx == GPIOA) RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
	else if (GPIOx == GPIOB) RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);
	else if (GPIOx == GPIOC) RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC, ENABLE);
	else if (GPIOx == GPIOD) RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOD, ENABLE);
	else if (GPIOx == GPIOE) RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOE, ENABLE);
	else if (GPIOx == GPIOF) RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOF, ENABLE);
	else if (GPIOx == GPIOG) RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOG, ENABLE);

	GPIO_InitTypeDef GPIO_Init_Struct;
	GPIO_Init_Struct.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init_Struct.GPIO_Mode = GPIO_Mode_Out_OD;
	GPIO_Init_Struct.GPIO_Pin = SCL | SDA;
	GPIO_Init(GPIOx, &GPIO_Init_Struct);

	GPIO_WriteBit(GPIOx, SCL, (BitAction)1);
	GPIO_WriteBit(GPIOx, SDA, (BitAction)1);
}

static void SI2C_Start(void)
{
	MyI2C_W_SDA(1);
	MyI2C_W_SCL(1);
	MyI2C_W_SDA(0);
	MyI2C_W_SCL(0);
}

static void SI2C_Stop(void)
{
	MyI2C_W_SDA(0);
	MyI2C_W_SCL(1);
	MyI2C_W_SDA(1);
}

static void SI2C_WriteByte(uint8_t Byte)
{
	uint8_t i;
	for (i = 0; i < 8; i++)
	{
		MyI2C_W_SDA(Byte & (0x80 >> i));
		Delay_us(2);
		MyI2C_W_SCL(1);
		Delay_us(2);
		MyI2C_W_SCL(0);
		Delay_us(2);
	}
}

static uint8_t SI2C_ReceiveByte(void)
{
	uint8_t i, Byte = 0x00;
	MyI2C_W_SDA(1);
	for (i = 0; i < 8; i++)
	{
		MyI2C_W_SCL(1);
		if (MyI2C_R_SDA() == 1) { Byte |= (0x80 >> i); }
		MyI2C_W_SCL(0);
	}
	return Byte;
}

static void SI2C_WriteAck(uint8_t AckBit)
{
	MyI2C_W_SDA(AckBit);
	MyI2C_W_SCL(1);
	MyI2C_W_SCL(0);
}

static uint8_t SI2C_ReceiveAck(void)
{
	uint8_t AckBit;
	MyI2C_W_SDA(1);
	MyI2C_W_SCL(1);
	AckBit = MyI2C_R_SDA();
	MyI2C_W_SCL(0);
	return AckBit;
}

void SI2C_WriteReg(uint8_t RegAddress, uint8_t Data)
{
	SI2C_Start();
	SI2C_WriteByte(I2C.I2C_Add << 1);
	SI2C_ReceiveAck();
	SI2C_WriteByte(RegAddress);
	SI2C_ReceiveAck();
	SI2C_WriteByte(Data);
	SI2C_ReceiveAck();
	SI2C_Stop();
}

uint8_t SI2C_ReadReg(uint8_t RegAddress)
{
	uint8_t Data;

	SI2C_Start();
	SI2C_WriteByte(I2C.I2C_Add << 1);
	SI2C_ReceiveAck();
	SI2C_WriteByte(RegAddress);
	SI2C_ReceiveAck();

	SI2C_Start();
	SI2C_WriteByte((I2C.I2C_Add << 1) | 0x01);
	SI2C_ReceiveAck();
	Data = SI2C_ReceiveByte();
	SI2C_WriteAck(1);
	SI2C_Stop();

	return Data;
}
