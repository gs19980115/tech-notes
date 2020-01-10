import chisel3._
import chisel3.util._

import chisel3.internal.InstanceId

case class FooParams (
    bar: Int = 1
)

trait HasFooParameters extends HasFooParameters {

}