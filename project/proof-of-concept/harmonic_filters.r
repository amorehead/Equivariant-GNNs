library(ggplot2)


m = 1 #rotation order


phi = seq(0,2*pi,length.out = 100)
r = seq(0,1,length.out = 50)

real_filter = function(r,phi,m)exp(-r^2)*cos(phi*m)
img_filter = function(r,phi,m)exp(-r^2)*sin(phi*m)



df = expand.grid(phi = phi, r = r)
df$real_filter = real_filter(df$r,df$phi,m)
df$img_filter = img_filter(df$r,df$phi,m)



df$input = 0
df$input[df$phi > 0 & df$phi < pi/6 & df$r>0.48 & df$r<0.52] = 1
df$input[df$phi > pi/12-0.01 & df$phi < pi/12+0.01 & df$r>0.3 & df$r<0.7] = 1

p_in <- ggplot(df) + geom_tile(aes(phi,r,fill=input))+ coord_polar(direction = -1,start=3*pi/2)

# plot the original image on the polar coordinates
plot(p_in)



df$input_rotate = 0
df$input_rotate[df$phi > 0+pi/2 & df$phi < pi/6+pi/2 & df$r>0.48 & df$r<0.52] = 1
df$input_rotate[df$phi > pi/12+pi/2-0.01 & df$phi < pi/12+0.01+pi/2 & df$r>0.3 & df$r<0.7] = 1
p_in_rotate <- ggplot(df) + geom_tile(aes(phi,r,fill=input_rotate))+ coord_polar(direction = -1,start=3*pi/2)


# plot the rotated image on the polar coordinates (pi/2, counter-clockwise)
plot(p_in_rotate)


# kernel weights
p_real <- ggplot(df) + geom_tile(aes(phi,r,fill=real_filter))+ coord_polar(direction = -1,start=3*pi/2)
p_img <- ggplot(df) + geom_tile(aes(phi,r,fill=img_filter))+ coord_polar(direction = -1,start=3*pi/2)


# Real and imaginary part of the convolution operation, which resulted in a complex number
out_real = sum(df$input*df$real_filter)
out_img = sum(df$input*df$img_filter)

out_real_rotate = sum(df$input_rotate*df$real_filter)
out_img_rotate = sum(df$input_rotate*df$img_filter)


plot(0,0,type='n',xlim = c(-100,100),ylim = c(-100,100))
abline(h=0,lty = 2)
abline(v=0,lty = 2)
lines(c(0,out_real),c(0,out_img),col=2)
lines(c(0,out_real_rotate),c(0,out_img_rotate),col=3)


