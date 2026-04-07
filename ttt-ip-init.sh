
#!/bin/sh

### BEGIN INIT INFO
# Provides:          deipce
# Short-Description: Load DE-IP Core Edge drivers
# Required-Start:    $local_fs
# Required-Stop:     $local_fs
# X-Start-Before:    networking
# Default-Start:     2 3 4 5
# Default-Stop
### END INIT INFO

pio_sysfs_name="c00f0f00.gpio"
fpga_eth_reset=""

# Print hardware serial number
print_serial()
{
    for i in {7..16}; do
        s=$(i2cget -y 0 0x51 $i b | sed 's/0x//')
        s_dec=$(bc <<< "obase=10;ibase=16;$s")
        printf \\$(printf '%o' $s_dec)
    done
}

# Get a MAC address from the EEPROM from a given offset.
get_mac()
{
    ret_mac=0
    for i in {1..6}; do
        o=$(($(i2cget -y 0 0x51 $i b)))
        ret_mac=$(($ret_mac *0x100 +$o))
    done
    ret_mac=$(($ret_mac + $1))
    # hex-print the mac address (12 chars = 6 octets in hex)
    # ... first sed puts ':' after every 2 chars, i.e., after every octet
    # ... second sed removes the last ':' trailing the line
    printf "%012x" $ret_mac | sed 's/\(.\{2\}\)/\1:/g' | sed 's/:$//'
}
 Configure minimum delays.
# Usage: set_delays PORT_NETDEV DELAY_NAME tx TX_DELAYS rx RX_DELAYS
set_min_delays()
{
    echo "$4" > "/sys/class/net/$1/phy/delay10tx_min"
    echo "$5" > "/sys/class/net/$1/phy/delay100tx_min"
    echo "$6" > "/sys/class/net/$1/phy/delay1000tx_min"

    echo "$8" > "/sys/class/net/$1/phy/delay10rx_min"
    echo "$9" > "/sys/class/net/$1/phy/delay100rx_min"
    echo "${10}" > "/sys/class/net/$1/phy/delay1000rx_min"
}

# Configure maximum delays.
# Usage: set_delays PORT_NETDEV DELAY_NAME tx TX_DELAYS rx RX_DELAYS
set_max_delays()
{
    echo "$4" > "/sys/class/net/$1/phy/delay10tx_max"
    echo "$5" > "/sys/class/net/$1/phy/delay100tx_max"
    echo "$6" > "/sys/class/net/$1/phy/delay1000tx_max"

    echo "$8" > "/sys/class/net/$1/phy/delay10rx_max"
    echo "$9" > "/sys/class/net/$1/phy/delay100rx_max"
    echo "${10}" > "/sys/class/net/$1/phy/delay1000rx_max"
}


start()
{
    echo "Loading drivers..."
    echo 1 > /proc/sys/kernel/printk
    # no eth0 when using DMA
    #ip link set eth0 up
    modprobe edgx-pfm_lkm

    echo -n "edgx_mdio-1:00" > /sys/class/net/sw0p2/phy/mdiobus
    echo -n "edgx_mdio-1:01" > /sys/class/net/sw0p3/phy/mdiobus
    echo -n "edgx_mdio-1:02" > /sys/class/net/sw0p4/phy/mdiobus
    echo -n "edgx_mdio-1:03" > /sys/class/net/sw0p5/phy/mdiobus

    ip link set dev sw0p1 address $(get_mac 3)
    ip link set dev sw0p2 address $(get_mac 4)
    ip link set dev sw0p3 address $(get_mac 5)
    ip link set dev sw0p4 address $(get_mac 6)
    ip link set dev sw0p5 address $(get_mac 7)
    ip link set dev sw0ep address $(get_mac 0)

    # set default mqprio setup for 3 traffic classes
    tc qdisc add dev sw0ep root mqprio num_tc 3 map 0 0 0 0 1 1 2 2 2 2 2 2 2 2 2 2 hw 1 mode channel
    tc qdisc replace dev sw0ep parent 8001:1 pfifo
    tc qdisc replace dev sw0ep parent 8001:2 pfifo
    tc qdisc replace dev sw0ep parent 8001:3 pfifo

    # Marvell 88E1510P - standard latency mode
    set_min_delays sw0p1 phydev tx 0 0 0 rx 0 0 0
    set_min_delays sw0p2 phydev tx 4032 412 109 rx 1083 220 203
    set_min_delays sw0p3 phydev tx 4032 412 109 rx 1083 220 203
    set_min_delays sw0p4 phydev tx 4032 412 109 rx 1083 220 203
    set_min_delays sw0p5 phydev tx 4032 412 109 rx 1083 220 203

    set_max_delays sw0p1 phydev tx 0 0 0 rx 0 0 0
    set_max_delays sw0p2 phydev tx 4832 492 135 rx 1183 220 211
    set_max_delays sw0p3 phydev tx 4832 492 135 rx 1183 220 211
    set_max_delays sw0p4 phydev tx 4832 492 135 rx 1183 220 211
    set_max_delays sw0p5 phydev tx 4832 492 135 rx 1183 220 211
}
stop()
{
        echo "Unloading drivers..."
        loaded="$(lsmod | awk '{ print $1 }')"
        echo "$loaded" | grep -q '^edgx_pfm_lkm$' && rmmod edgx_pfm_lkm
}

case "$1" in
    start|restart|force-reload)
        echo "Configuring FPGA ..."
        echo "  Hardware S/N: "$(print_serial)
        start
	;;
    stop)
	stop
        ;;
esac

exit 0

