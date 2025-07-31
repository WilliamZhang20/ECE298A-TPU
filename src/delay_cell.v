module buffer (
    input wire A,
    output wire X
);
    assign #1 X = A;
endmodule