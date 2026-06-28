#ifndef __MYI2C_H
#define __MYI2C_H

#include "stm32f10x.h"
#include "Delay.h"

void SI2C_Init(GPIO_TypeDef* GPIOx, uint16_t SCL, uint16_t SDA, uint16_t I2C_Address);
void SI2C_WriteReg(uint8_t RegAddress, uint8_t Data);
uint8_t SI2C_ReadReg(uint8_t RegAddress);

#endif
