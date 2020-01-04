package demo

import chisel3._
import chisel3.util._

import freechips.rocketchip.util.{uintToBitPat,UIntIsOneOf}

// 我们希望通过键值key，根据译码表译码得到三个控制信号(a,b,c)
// 若译码表中找不到key,则置默认值
//  - DecodeTable:  译码表
//  - CtrlSigs:     译码得到的控制信号
//  - DecodeDemo:   顶层demo模块
//  - DecodeDemoDriver: 生成verilog (runMain demo.DecodeDemoDriver)

class CtrlSigs extends Bundle {

    // 定义译码得到的控制信号a、b、c,放在一个Bundle中
    val a = UInt(2.W)
    val b = UInt(2.W)
    val c = UInt(2.W)

    // 默认信号值 均为0
    def default: List[BitPat] =
        List(0.U(2.W), 0.U(2.W), 0.U(2.W))

    // 译码逻辑，调用rocketchip中的DecodeLogic,译码出结果，并赋值给该Bundle中的a、b、c
    def decode(key: UInt, table: Iterable[(BitPat, List[BitPat])]) = {
        val decoder = freechips.rocketchip.rocket.DecodeLogic(key, default, table)
        val sigs = Seq(a, b, c)
        sigs zip decoder map {case(s,d) => s := d}
        this
    }
}

class DecodeTable {
    val table: Array[(BitPat, List[BitPat])] = Array(
        BitPat("b01") -> List(0.U(2.W), 1.U(2.W), 2.U(2.W)),
        BitPat("b11") -> List(1.U(2.W), 2.U(2.W), 0.U(2.W)),
        BitPat("b10") -> List(2.U(2.W), 0.U(2.W), 1.U(2.W))
    )
}

class DecodeDemo extends Module {
    val io = IO(new Bundle{
        val key = Input(UInt(2.W))
        val ctrl_sigs = Output(new CtrlSigs)
    })

    val decode_table = (new DecodeTable).table
    io.ctrl_sigs := Wire(new CtrlSigs).decode(io.key, decode_table)

}


object DecodeDemoDriver extends App {
  chisel3.Driver.execute(args, () => new DecodeDemo)
}
