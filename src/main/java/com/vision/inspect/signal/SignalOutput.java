package com.vision.inspect.signal;

/**
 * OK/NG 信号输出抽象：可实现 GPIO、继电器、PLC Modbus、MES 上报等。
 */
public interface SignalOutput {
    void outputOk();

    void outputNg();

    void reset();
}
