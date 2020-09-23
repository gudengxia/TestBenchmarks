﻿/**************************************************************************
 * Alg.h (Version 1.1.1) created on May 12, 2019.
 *************************************************************************/

/**************************************************************************
 * 说明：
 *		1 本程序实现了随机数生成接口，即
        aigis_rand_get_sd_byts（）
		aigis_rand_init（）
		aigis_rand_byts（）
		若使用外部实现，请用相关实现替换randombytes.c文件中三个函数的实现。
 *************************************************************************/

#ifndef ALG_H
#define ALG_H

//#define PKC_ALG_API __declspec(dllexport)
#define PKC_ALG_API __attribute__((visibility("default"))) //fzhang

typedef unsigned char* puchar_t;
typedef unsigned long long puchar_byts_t;

/**************************************************************************
 * 函数：aigis_sig_get_pk_byts
 * 功能：获取数字签名算法声称公钥（字节）长度，必须与算法设计文档一致
 * 返回：数字签名算法声称公钥（字节）长度
 *************************************************************************/
PKC_ALG_API puchar_byts_t aigis_sig_get_pk_byts();

/**************************************************************************
 * 函数：aigis_sig_get_sk_byts
 * 功能：获取数字签名算法声称私钥（字节）长度，必须与算法设计文档一致
 * 返回：数字签名算法声称私钥（字节）长度
 *************************************************************************/
PKC_ALG_API puchar_byts_t aigis_sig_get_sk_byts();

/**************************************************************************
 * 函数：aigis_sig_get_sn_byts
 * 功能：获取数字签名算法声称签名（字节）长度，必须与算法设计文档一致
 * 返回：数字签名算法声称签名（字节）长度
 *************************************************************************/
PKC_ALG_API puchar_byts_t aigis_sig_get_sn_byts();

/**************************************************************************
 * 函数：sig_keygen
 * 功能：数字签名——密钥生成算法
 * 输出：pk：公钥
 *		 pk_byts：公钥字节长度
 *		 sk：私钥
 *		 sk_byts：私钥字节长度
 * 返回：成功执行返回0，否则返回错误代码（负数）
 *************************************************************************/
PKC_ALG_API int aigis_sig_keygen(
	puchar_t pk, puchar_byts_t* pk_byts, 
	puchar_t sk, puchar_byts_t* sk_byts);

/**************************************************************************
 * 函数：sig_sign
 * 功能：数字签名——签名算法
 * 输入：sk：私钥
 *		 sk_byts：私钥字节长度
 *		 m：消息
 *		 m_byts：消息字节长度
 * 输出：sn：签名
 *		 sn_byts：签名字节长度
 * 返回：成功执行返回0，否则返回错误代码（负数）
 *************************************************************************/
PKC_ALG_API int aigis_sig_sign(
	puchar_t sk, puchar_byts_t sk_byts,
	puchar_t m, puchar_byts_t m_byts,
	puchar_t sn, puchar_byts_t* sn_byts);

/**************************************************************************
 * 函数：sig_verf
 * 功能：数字签名——验证算法
 * 输入：pk：公钥
 *		 pk_byts：公钥字节长度
 *		 sn：签名
 *		 sn_byts：签名字节长度
 *		 m：消息
 *		 m_byts：消息字节长度
 * 返回：成功执行返回1（验证通过）或0（验证不通过），否则返回错误代码（负
 *		 数）
 *************************************************************************/
PKC_ALG_API int aigis_sig_verf(
	puchar_t pk, puchar_byts_t pk_byts,
	puchar_t sn, puchar_byts_t sn_byts,
	puchar_t m, puchar_byts_t m_byts);

#endif