#include "stm32f10x.h"
#include "Delay.h"
#include "MyI2C.h"
#include "BME280.h"
#include "stdio.h"

/* USART1: PA9=TX, PA10=RX */
void USART1_Init(uint32_t baud)
{
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA | RCC_APB2Periph_USART1, ENABLE);

	GPIO_InitTypeDef gpio;
	gpio.GPIO_Pin   = GPIO_Pin_9;
	gpio.GPIO_Mode  = GPIO_Mode_AF_PP;
	gpio.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &gpio);

	gpio.GPIO_Pin   = GPIO_Pin_10;
	gpio.GPIO_Mode  = GPIO_Mode_IN_FLOATING;
	GPIO_Init(GPIOA, &gpio);

	USART_InitTypeDef usart;
	usart.USART_BaudRate            = baud;
	usart.USART_WordLength          = USART_WordLength_8b;
	usart.USART_StopBits            = USART_StopBits_1;
	usart.USART_Parity              = USART_Parity_No;
	usart.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	usart.USART_Mode                = USART_Mode_Tx | USART_Mode_Rx;
	USART_Init(USART1, &usart);
	USART_Cmd(USART1, ENABLE);
}

int fputc(int ch, FILE *f)
{
	while (USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET);
	USART_SendData(USART1, (uint8_t)ch);
	return ch;
}

/*
 * MicroLIB doesn't support %f. Print a double as int.fraction
 * e.g. 25.37 → "25.37"
 */
static void print_double_2dp(double val)
{
	int sign = 0;
	if (val < 0) { sign = 1; val = -val; }
	int int_part = (int)val;
	int frac     = (int)((val - int_part) * 100.0 + 0.5);
	if (frac >= 100) { int_part++; frac = 0; }
	if (sign) putchar('-');
	printf("%d.%02d", int_part, frac);
}

int main(void)
{
	Delay_us(1);
	USART1_Init(115200);

	printf("\r\n=== BME280 Sensor Demo (STM32F103C8T6) ===\r\n");

	BME280_Init(GPIOB, GPIO_Pin_6, GPIO_Pin_7);

	uint8_t id = BME280_Get_ID();
	printf("Chip ID: 0x%02X (expected: 0x60)\r\n\r\n", id);
	if (id != 0x60 && id != 0x58)
	{
		printf("ERROR: Sensor not found! Check wiring.\r\n");
		while (1);
	}

	while (1)
	{
		double temp = BME280_Get_Temp();
		double hum  = BME280_Get_Hum();
		double pres = BME280_Get_Press();

		/* Single-line output — matches Python regex */
		printf("Temperature: ");
		print_double_2dp(temp);
		printf(" C  |  Humidity: ");
		print_double_2dp(hum);
		printf(" %%  |  Pressure: ");
		print_double_2dp(pres);
		printf(" Pa\r\n");

		Delay_s(1);
	}
}
